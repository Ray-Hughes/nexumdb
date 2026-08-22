// Hide the console window on Windows release builds; a GUI app spawning a
// terminal behind itself looks broken.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    nexum_viewer_lib::run()
}
