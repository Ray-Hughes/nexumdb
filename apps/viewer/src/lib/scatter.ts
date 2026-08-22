/**
 * A minimal WebGL2 point renderer for the embedding projection.
 *
 * Written in-tree rather than pulled from deck.gl or regl: the whole
 * requirement is "draw N coloured points with pan and zoom", which is one
 * vertex shader and one fragment shader. A general-purpose viz library would
 * add megabytes to the bundle to solve a problem this file solves in a page.
 *
 * Falls back to `null` if WebGL2 is unavailable, so the caller can render a
 * canvas-2D version rather than a blank rectangle.
 */

const VERTEX_SHADER = `#version 300 es
in vec2 a_position;
in vec3 a_color;
uniform vec2 u_pan;
uniform float u_zoom;
uniform vec2 u_resolution;
uniform float u_pointSize;
out vec3 v_color;

void main() {
  // World space is roughly -1..1 on both axes; fit it to the shorter screen
  // edge so the layout keeps its aspect ratio.
  float scale = min(u_resolution.x, u_resolution.y) * 0.45 * u_zoom;
  vec2 screen = (a_position * scale) + u_pan + (u_resolution * 0.5);
  vec2 clip = (screen / u_resolution) * 2.0 - 1.0;
  gl_Position = vec4(clip.x, -clip.y, 0.0, 1.0);
  gl_PointSize = u_pointSize;
  v_color = a_color;
}`;

const FRAGMENT_SHADER = `#version 300 es
precision mediump float;
in vec3 v_color;
out vec4 outColor;

void main() {
  // Round points with a soft edge; square points read as artefacts.
  vec2 offset = gl_PointCoord - vec2(0.5);
  float distance = length(offset);
  if (distance > 0.5) discard;
  float alpha = smoothstep(0.5, 0.42, distance);
  outColor = vec4(v_color, alpha);
}`;

export interface ScatterHandle {
  /** Upload new geometry. Colours are 0..1 RGB triples. */
  setData(positions: Float32Array, colors: Float32Array): void;
  /** Redraw at the given view. */
  draw(view: { panX: number; panY: number; zoom: number; pointSize: number }): void;
  /** Match the drawing buffer to the element size and device pixel ratio. */
  resize(width: number, height: number, dpr: number): void;
  dispose(): void;
  count: number;
}

function compile(gl: WebGL2RenderingContext, type: number, source: string) {
  const shader = gl.createShader(type);
  if (!shader) throw new Error("could not create shader");
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(shader);
    gl.deleteShader(shader);
    throw new Error(`shader failed to compile: ${log}`);
  }
  return shader;
}

export function createScatter(canvas: HTMLCanvasElement): ScatterHandle | null {
  const gl = canvas.getContext("webgl2", {
    antialias: true,
    alpha: true,
    premultipliedAlpha: false,
  });
  if (!gl) return null;

  let program: WebGLProgram;
  try {
    const vertex = compile(gl, gl.VERTEX_SHADER, VERTEX_SHADER);
    const fragment = compile(gl, gl.FRAGMENT_SHADER, FRAGMENT_SHADER);
    const created = gl.createProgram();
    if (!created) return null;
    program = created;
    gl.attachShader(program, vertex);
    gl.attachShader(program, fragment);
    gl.linkProgram(program);
    gl.deleteShader(vertex);
    gl.deleteShader(fragment);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      gl.deleteProgram(program);
      return null;
    }
  } catch {
    return null;
  }

  const vao = gl.createVertexArray();
  const positionBuffer = gl.createBuffer();
  const colorBuffer = gl.createBuffer();
  if (!vao || !positionBuffer || !colorBuffer) return null;

  const positionLocation = gl.getAttribLocation(program, "a_position");
  const colorLocation = gl.getAttribLocation(program, "a_color");
  const panLocation = gl.getUniformLocation(program, "u_pan");
  const zoomLocation = gl.getUniformLocation(program, "u_zoom");
  const resolutionLocation = gl.getUniformLocation(program, "u_resolution");
  const pointSizeLocation = gl.getUniformLocation(program, "u_pointSize");

  gl.bindVertexArray(vao);
  gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
  gl.enableVertexAttribArray(positionLocation);
  gl.vertexAttribPointer(positionLocation, 2, gl.FLOAT, false, 0, 0);
  gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
  gl.enableVertexAttribArray(colorLocation);
  gl.vertexAttribPointer(colorLocation, 3, gl.FLOAT, false, 0, 0);
  gl.bindVertexArray(null);

  gl.enable(gl.BLEND);
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

  const handle: ScatterHandle = {
    count: 0,

    setData(positions, colors) {
      gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);
      gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, colors, gl.STATIC_DRAW);
      handle.count = positions.length / 2;
    },

    resize(width, height, dpr) {
      canvas.width = Math.max(1, Math.floor(width * dpr));
      canvas.height = Math.max(1, Math.floor(height * dpr));
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;
      gl.viewport(0, 0, canvas.width, canvas.height);
    },

    draw({ panX, panY, zoom, pointSize }) {
      gl.clearColor(0, 0, 0, 0);
      gl.clear(gl.COLOR_BUFFER_BIT);
      if (handle.count === 0) return;

      gl.useProgram(program);
      gl.bindVertexArray(vao);
      const dpr = canvas.width / Math.max(1, parseFloat(canvas.style.width || "1"));
      gl.uniform2f(panLocation, panX * dpr, panY * dpr);
      gl.uniform1f(zoomLocation, zoom);
      gl.uniform2f(resolutionLocation, canvas.width, canvas.height);
      gl.uniform1f(pointSizeLocation, pointSize * dpr);
      gl.drawArrays(gl.POINTS, 0, handle.count);
      gl.bindVertexArray(null);
    },

    dispose() {
      gl.deleteBuffer(positionBuffer);
      gl.deleteBuffer(colorBuffer);
      gl.deleteVertexArray(vao);
      gl.deleteProgram(program);
    },
  };

  return handle;
}

/** Convert `#rrggbb` to a 0..1 RGB triple. */
export function hexToRgb(hex: string): [number, number, number] {
  const value = hex.replace("#", "");
  const full =
    value.length === 3
      ? value
          .split("")
          .map((c) => c + c)
          .join("")
      : value;
  return [
    parseInt(full.slice(0, 2), 16) / 255,
    parseInt(full.slice(2, 4), 16) / 255,
    parseInt(full.slice(4, 6), 16) / 255,
  ];
}

/**
 * A stable, well-spread colour for the nth distinct series.
 *
 * Uses the golden-angle hue rotation so adjacent documents get distant hues
 * without maintaining a palette, and keeps saturation and lightness fixed so
 * every colour reads at the same weight on a dark background.
 */
export function colorForKey(index: number): [number, number, number] {
  const hue = (index * 137.508) % 360;
  return hslToRgb(hue / 360, 0.58, 0.62);
}

function hslToRgb(h: number, s: number, l: number): [number, number, number] {
  if (s === 0) return [l, l, l];
  const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
  const p = 2 * l - q;
  const channel = (t: number) => {
    let value = t;
    if (value < 0) value += 1;
    if (value > 1) value -= 1;
    if (value < 1 / 6) return p + (q - p) * 6 * value;
    if (value < 1 / 2) return q;
    if (value < 2 / 3) return p + (q - p) * (2 / 3 - value) * 6;
    return p;
  };
  return [channel(h + 1 / 3), channel(h), channel(h - 1 / 3)];
}
