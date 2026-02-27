// Type declarations for mapbox modules that don't have types
declare module '@mapbox/point-geometry' {
  export default class Point {
    x: number;
    y: number;
    constructor(x: number, y: number);
    clone(): Point;
    add(p: Point): Point;
    sub(p: Point): Point;
    mult(k: number): Point;
    div(k: number): Point;
    rotate(angle: number): Point;
    rotateAround(angle: number, p: Point): Point;
    matMult(m: [number, number, number, number]): Point;
    unit(): Point;
    perp(): Point;
    round(): Point;
    mag(): number;
    equals(p: Point): boolean;
    dist(p: Point): number;
    distSqr(p: Point): number;
    angle(): number;
    angleTo(p: Point): number;
    angleWith(p: Point): number;
    angleWithSep(x: number, y: number): number;
    static convert(p: Point | [number, number]): Point;
  }
}

declare module '@mapbox/unitbezier' {
  export default class UnitBezier {
    constructor(p1x: number, p1y: number, p2x: number, p2y: number);
    sampleCurveX(t: number): number;
    sampleCurveY(t: number): number;
    sampleCurveDerivativeX(t: number): number;
    solveCurveX(x: number, epsilon?: number): number;
    solve(x: number, epsilon?: number): number;
  }
}
