/*
  main.dart
  ---------
  Hello World proof-of-concept #3 — Desktop Context Monitor.
  MSc Thesis — Dimitris Moforis, University of Piraeus, Dept. of Digital Systems

  What this app does:
    1. Runs as a Flutter window on Windows
    2. Every second, samples:
         - Active window title via Windows API (GetForegroundWindow)
         - User idle time via Windows API (GetLastInputInfo) — ms since last input
         - Window switch count
    3. Every WINDOW_SECONDS (30s) saves a reading to stress_monitor.db
    4. Shows a live dashboard

  Architecture decision — why no global hooks:
    Low-level Windows hooks (SetWindowsHookEx WH_KEYBOARD_LL) fire on a
    background OS thread, which cannot call Dart/Flutter code directly.
    This causes the "Cannot invoke native callback outside an isolate" crash.
    
    Instead we use GetLastInputInfo() — a safe polling API that returns the
    timestamp of the last keyboard OR mouse event system-wide. From this we
    derive idle time, which is a reliable proxy for activity level.
    This approach is used in production monitoring tools and is entirely
    appropriate for a 30-second window stress detection system.

  What GetLastInputInfo gives us:
    - idle_seconds: seconds since last keyboard/mouse input anywhere on system
    - activity_pct: % of the window where the user was active (not idle)
    These replace raw keystroke/click counts as activity intensity proxies.

  Dependencies:
    flutter pub add sqflite_common_ffi win32
*/

import 'dart:async';
import 'dart:ffi';
import 'dart:io';
import 'package:ffi/ffi.dart';
import 'package:flutter/material.dart';
import 'package:sqflite_common_ffi/sqflite_ffi.dart';
import 'package:win32/win32.dart';

// ─────────────────────────────────────────────────────────────────────────────
// CONFIGURATION
// ─────────────────────────────────────────────────────────────────────────────

// Derive DB path from working directory (set by run_all.py to project root).
// Falls back gracefully with a clear error if the directory does not exist.
String _resolveDbPath() {
  final cwd = Directory.current.path;
  final candidate = '$cwd\\data\\stress_monitor.db'
      .replaceAll('\\', Platform.pathSeparator)
      .replaceAll('/', Platform.pathSeparator);
  return candidate;
}

final String dbPath = _resolveDbPath();

const int WINDOW_SECONDS = 30; // seconds per measurement window

// Idle threshold: if no input for more than this many seconds, user is idle
const int IDLE_THRESHOLD_SECONDS = 5;

// ─────────────────────────────────────────────────────────────────────────────
// WINDOWS API HELPERS
// Safe polling functions — called from Dart's main thread, no threading issues
// ─────────────────────────────────────────────────────────────────────────────

/// Returns the title of the currently focused window.
/// Uses GetForegroundWindow + GetWindowText from the win32 package.
String getActiveWindowTitle() {
  final hwnd = GetForegroundWindow();
  if (hwnd == 0) return 'Unknown';
  final buffer = wsalloc(256);
  try {
    GetWindowText(hwnd, buffer, 256);
    final title = buffer.toDartString();
    return title.isEmpty ? 'Unknown' : title;
  } finally {
    free(buffer);
  }
}

/// Returns how many seconds have passed since the last keyboard or mouse input.
/// Uses GetLastInputInfo — a safe, polling-based Windows API.
/// This works system-wide — it detects input in ANY application.
double getIdleSeconds() {
  // LASTINPUTINFO struct: cbSize (uint32) + dwTime (uint32)
  final info = calloc<LASTINPUTINFO>();
  try {
    info.ref.cbSize = sizeOf<LASTINPUTINFO>();
    final ok = GetLastInputInfo(info);
    if (ok == FALSE) return 0.0;

    // GetTickCount() returns milliseconds since system boot
    final now = GetTickCount();
    final lastInput = info.ref.dwTime;

    // Handle tick count wraparound (occurs every ~49 days of uptime)
    final elapsedMs = (now - lastInput) & 0xFFFFFFFF;
    return elapsedMs / 1000.0;
  } finally {
    free(info);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// WINDOW CATEGORY CLASSIFIER
// ─────────────────────────────────────────────────────────────────────────────

String classifyWindow(String title) {
  final t = title.toLowerCase();
  if (t.contains('visual studio') ||
      t.contains('code') ||
      t.contains('pycharm') ||
      t.contains('terminal') ||
      t.contains('powershell') ||
      t.contains('cmd') ||
      t.contains('android studio'))
    return 'IDE/Terminal';
  if (t.contains('teams') ||
      t.contains('slack') ||
      t.contains('zoom') ||
      t.contains('meet') ||
      t.contains('outlook') ||
      t.contains('gmail') ||
      t.contains('mail'))
    return 'Communication';
  if (t.contains('word') ||
      t.contains('excel') ||
      t.contains('powerpoint') ||
      t.contains('notion') ||
      t.contains('obsidian') ||
      t.contains('docs'))
    return 'Document';
  if (t.contains('chrome') ||
      t.contains('firefox') ||
      t.contains('edge') ||
      t.contains('safari'))
    return 'Browser';
  if (t.contains('spotify') ||
      t.contains('netflix') ||
      t.contains('youtube') ||
      t.contains('vlc'))
    return 'Media';
  return 'Other';
}

// ─────────────────────────────────────────────────────────────────────────────
// DATABASE LAYER
// ─────────────────────────────────────────────────────────────────────────────

Future<Database> initDatabase() async {
  sqfliteFfiInit();
  final db = await databaseFactoryFfi.openDatabase(
    dbPath,
    options: OpenDatabaseOptions(
      version: 1,
      onOpen: (db) async {
        // Create table if not already present (Python scripts create the DB first)
        await db.execute('''
          CREATE TABLE IF NOT EXISTS desktop_readings (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp        TEXT    NOT NULL,
            active_window    TEXT,
            app_category     TEXT,
            idle_seconds     REAL,
            activity_pct     REAL,
            window_switches  INTEGER
          )
        ''');
      },
    ),
  );
  print('[DB] desktop_readings table ready → $dbPath');
  return db;
}

Future<void> saveReading({
  required Database db,
  required String activeWindow,
  required String appCategory,
  required double idleSeconds,
  required double activityPct,
  required int windowSwitches,
}) async {
  final ts = DateTime.now().toIso8601String().substring(0, 19);
  await db.insert('desktop_readings', {
    'timestamp': ts,
    'active_window': activeWindow,
    'app_category': appCategory,
    'idle_seconds': idleSeconds,
    'activity_pct': activityPct,
    'window_switches': windowSwitches,
  });
  print(
    '[DB] Saved → $ts | $appCategory | '
    'Idle: ${idleSeconds.toStringAsFixed(1)}s | '
    'Active: ${activityPct.toStringAsFixed(0)}% | '
    'Switches: $windowSwitches',
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// DESKTOP MONITOR — CORE LOGIC
// ─────────────────────────────────────────────────────────────────────────────

class DesktopMonitor {
  // Per-window accumulators
  int _windowSwitches = 0;
  int _idleFrames = 0; // seconds where user was idle this window
  int _activeFrames = 0; // seconds where user was active this window
  String _lastWindow = '';
  String _currentWindow = '';
  int _secondsElapsed = 0;

  /// Called every second. Samples all metrics and returns live display values.
  Map<String, dynamic> tick() {
    _secondsElapsed++;

    // Sample active window
    _currentWindow = getActiveWindowTitle();
    if (_currentWindow != _lastWindow && _lastWindow.isNotEmpty) {
      _windowSwitches++;
    }
    _lastWindow = _currentWindow;

    // Sample idle time from Windows API
    final idleSecs = getIdleSeconds();
    if (idleSecs >= IDLE_THRESHOLD_SECONDS) {
      _idleFrames++; // this second counts as idle
    } else {
      _activeFrames++; // this second counts as active
    }

    // Activity percentage so far in this window
    final total = _idleFrames + _activeFrames;
    final activityPct = total > 0 ? (_activeFrames / total * 100) : 100.0;

    return {
      'window': _currentWindow,
      'category': classifyWindow(_currentWindow),
      'idleSecs': idleSecs, // current live idle duration
      'activityPct': activityPct, // % active so far this window
      'switches': _windowSwitches,
      'remaining': WINDOW_SECONDS - _secondsElapsed,
    };
  }

  /// Compute summary for the completed window and reset accumulators.
  Map<String, dynamic> getReadingAndReset() {
    final total = _idleFrames + _activeFrames;
    final activityPct = total > 0 ? (_activeFrames / total * 100.0) : 100.0;
    final totalIdle = _idleFrames.toDouble();

    final reading = {
      'active_window': _currentWindow,
      'app_category': classifyWindow(_currentWindow),
      'idle_seconds': totalIdle,
      'activity_pct': activityPct,
      'window_switches': _windowSwitches,
    };

    // Reset for next window
    _windowSwitches = 0;
    _idleFrames = 0;
    _activeFrames = 0;
    _secondsElapsed = 0;

    return reading;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// FLUTTER UI
// ─────────────────────────────────────────────────────────────────────────────

void main() async {
  WidgetsFlutterBinding.ensureInitialized();

  // Validate that the data/ directory exists before attempting to open the DB.
  // A missing directory means Flutter was not launched from the project root.
  final dbDir = Directory(dbPath).parent;
  if (!dbDir.existsSync()) {
    print('[Flutter] ERROR: Database directory not found at $dbPath');
    print('[Flutter] Ensure Flutter is launched from the project root via run_all.py');
  }

  final db = await initDatabase();
  runApp(DesktopMonitorApp(db: db));
}

class DesktopMonitorApp extends StatelessWidget {
  final Database db;
  const DesktopMonitorApp({super.key, required this.db});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Desktop Monitor — Stress Detection PoC #3',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.teal),
        useMaterial3: true,
      ),
      home: MonitorScreen(db: db),
    );
  }
}

class MonitorScreen extends StatefulWidget {
  final Database db;
  const MonitorScreen({super.key, required this.db});

  @override
  State<MonitorScreen> createState() => _MonitorScreenState();
}

class _MonitorScreenState extends State<MonitorScreen> {
  final DesktopMonitor _monitor = DesktopMonitor();

  // Live display values
  String _window = 'Waiting...';
  String _category = '—';
  double _idleSecs = 0;
  double _activityPct = 100;
  int _switches = 0;
  int _remaining = WINDOW_SECONDS;
  int _readingCount = 0;
  String _lastSaved = '—';

  late Timer _ticker;

  @override
  void initState() {
    super.initState();

    // Tick every second on Flutter's main thread — safe, no FFI threading issues
    _ticker = Timer.periodic(const Duration(seconds: 1), (_) async {
      final live = _monitor.tick();

      setState(() {
        _window = live['window'];
        _category = live['category'];
        _idleSecs = live['idleSecs'];
        _activityPct = live['activityPct'];
        _switches = live['switches'];
        _remaining = (live['remaining'] as int).clamp(0, WINDOW_SECONDS);
      });

      // Save reading when window completes
      if (live['remaining'] <= 0) {
        final r = _monitor.getReadingAndReset();
        await saveReading(
          db: widget.db,
          activeWindow: r['active_window'],
          appCategory: r['app_category'],
          idleSeconds: r['idle_seconds'],
          activityPct: r['activity_pct'],
          windowSwitches: r['window_switches'],
        );
        setState(() {
          _readingCount++;
          _lastSaved = DateTime.now().toIso8601String().substring(11, 19);
        });
      }
    });
  }

  @override
  void dispose() {
    _ticker.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // ── Header ────────────────────────────────────────────────────
            const Text(
              'Desktop Context Monitor',
              style: TextStyle(fontSize: 22, fontWeight: FontWeight.w500),
            ),
            const Text(
              'MSc Thesis · PoC #3 · Dimitris Moforis',
              style: TextStyle(fontSize: 13, color: Colors.grey),
            ),
            const SizedBox(height: 24),

            // ── Countdown bar ─────────────────────────────────────────────
            Text(
              'Next reading in $_remaining seconds',
              style: const TextStyle(fontSize: 13, color: Colors.grey),
            ),
            const SizedBox(height: 6),
            LinearProgressIndicator(
              value: 1 - (_remaining / WINDOW_SECONDS),
              backgroundColor: Colors.grey.shade200,
              color: Colors.teal,
            ),
            const SizedBox(height: 24),

            // ── Live metrics ──────────────────────────────────────────────
            Wrap(
              spacing: 12,
              runSpacing: 12,
              children: [
                _MetricCard(
                  label: 'Active window',
                  value: _window,
                  color: Colors.teal.shade50,
                ),
                _MetricCard(
                  label: 'Category',
                  value: _category,
                  color: Colors.blue.shade50,
                ),
                _MetricCard(
                  label: 'Current idle time',
                  value: '${_idleSecs.toStringAsFixed(1)}s',
                  color: _idleSecs >= IDLE_THRESHOLD_SECONDS
                      ? Colors
                            .red
                            .shade50 // highlight when idle
                      : Colors.green.shade50,
                ), // green when active
                _MetricCard(
                  label: 'Activity this window',
                  value: '${_activityPct.toStringAsFixed(0)}%',
                  color: Colors.orange.shade50,
                ),
                _MetricCard(
                  label: 'Window switches',
                  value: '$_switches',
                  color: Colors.purple.shade50,
                ),
              ],
            ),
            const SizedBox(height: 24),

            // ── Readings counter ──────────────────────────────────────────
            if (_readingCount > 0) ...[
              const Divider(),
              const SizedBox(height: 12),
              Text(
                'Readings saved: $_readingCount',
                style: const TextStyle(fontWeight: FontWeight.w500),
              ),
              Text(
                'Last saved at: $_lastSaved',
                style: const TextStyle(color: Colors.grey, fontSize: 13),
              ),
            ],

            const Spacer(),

            // ── DB path footer ─────────────────────────────────────────────
            Text(
              'Saving to: $dbPath',
              style: const TextStyle(fontSize: 11, color: Colors.grey),
              overflow: TextOverflow.ellipsis,
            ),
          ],
        ),
      ),
    );
  }
}

/// Reusable metric display card.
class _MetricCard extends StatelessWidget {
  final String label;
  final String value;
  final Color color;
  const _MetricCard({
    required this.label,
    required this.value,
    required this.color,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      width: 200,
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: color,
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: Colors.grey.shade300, width: 0.5),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(label, style: const TextStyle(fontSize: 11, color: Colors.grey)),
          const SizedBox(height: 4),
          Text(
            value,
            style: const TextStyle(fontSize: 15, fontWeight: FontWeight.w500),
            overflow: TextOverflow.ellipsis,
          ),
        ],
      ),
    );
  }
}
