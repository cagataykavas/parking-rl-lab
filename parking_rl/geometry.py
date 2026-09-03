from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    heading_deg: float


@dataclass(frozen=True)
class VehicleFootprint:
    length: float
    width: float

    def __post_init__(self) -> None:
        if self.length <= 0 or self.width <= 0:
            raise ValueError("vehicle dimensions must be positive")


@dataclass(frozen=True)
class ParkingSlot:
    center_x: float
    center_y: float
    heading_deg: float
    length: float
    width: float

    def __post_init__(self) -> None:
        if self.length <= 0 or self.width <= 0:
            raise ValueError("slot dimensions must be positive")


@dataclass(frozen=True)
class AxisAlignedRect:
    center_x: float
    center_y: float
    width: float
    height: float

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("rectangle dimensions must be positive")

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return (
            self.center_x - self.width / 2,
            self.center_x + self.width / 2,
            self.center_y - self.height / 2,
            self.center_y + self.height / 2,
        )


@dataclass(frozen=True)
class ParkingQuality:
    center_distance: float
    heading_error_deg: float
    inside_fraction: float
    min_slot_clearance: float
    obstacle_clearance: float
    world_clearance: float
    fully_inside_slot: bool
    collision: bool

    @property
    def pose_score(self) -> float:
        distance_term = math.exp(-self.center_distance / 1.5)
        heading_term = math.exp(-self.heading_error_deg / 12.0)
        containment_term = self.inside_fraction
        return float((distance_term + heading_term + containment_term) / 3.0)


def wrap_angle_deg(angle: float) -> float:
    return (float(angle) + 180.0) % 360.0 - 180.0


def rotation_matrix(heading_deg: float) -> np.ndarray:
    angle = math.radians(float(heading_deg))
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return np.asarray([[cosine, -sine], [sine, cosine]], dtype=float)


def local_to_world(points: np.ndarray, pose: Pose2D) -> np.ndarray:
    materialized = np.asarray(points, dtype=float)
    if materialized.ndim != 2 or materialized.shape[1] != 2:
        raise ValueError("points must have shape (n, 2)")
    rotated = materialized @ rotation_matrix(pose.heading_deg).T
    rotated[:, 0] += pose.x
    rotated[:, 1] += pose.y
    return rotated


def world_to_local(points: np.ndarray, pose: Pose2D) -> np.ndarray:
    materialized = np.asarray(points, dtype=float)
    if materialized.ndim != 2 or materialized.shape[1] != 2:
        raise ValueError("points must have shape (n, 2)")
    translated = materialized - np.asarray([pose.x, pose.y], dtype=float)
    return translated @ rotation_matrix(-pose.heading_deg).T


def vehicle_corners(pose: Pose2D, footprint: VehicleFootprint) -> np.ndarray:
    half_length = footprint.length / 2.0
    half_width = footprint.width / 2.0
    local = np.asarray(
        [
            [half_length, half_width],
            [half_length, -half_width],
            [-half_length, -half_width],
            [-half_length, half_width],
        ],
        dtype=float,
    )
    return local_to_world(local, pose)


def slot_corners(slot: ParkingSlot) -> np.ndarray:
    return vehicle_corners(
        Pose2D(slot.center_x, slot.center_y, slot.heading_deg),
        VehicleFootprint(slot.length, slot.width),
    )


def point_in_convex_polygon(point: np.ndarray, polygon: np.ndarray, *, eps: float = 1e-9) -> bool:
    p = np.asarray(point, dtype=float)
    poly = np.asarray(polygon, dtype=float)
    if p.shape != (2,) or poly.ndim != 2 or poly.shape[1] != 2 or len(poly) < 3:
        raise ValueError("invalid point or polygon")

    signs: list[float] = []
    for index in range(len(poly)):
        start = poly[index]
        end = poly[(index + 1) % len(poly)]
        edge = end - start
        relative = p - start
        signs.append(float(edge[0] * relative[1] - edge[1] * relative[0]))
    non_zero = [value for value in signs if abs(value) > eps]
    return not non_zero or all(value >= 0 for value in non_zero) or all(value <= 0 for value in non_zero)


def _project_polygon(axis: np.ndarray, polygon: np.ndarray) -> tuple[float, float]:
    projection = polygon @ axis
    return float(np.min(projection)), float(np.max(projection))


def _polygon_axes(polygon: np.ndarray) -> list[np.ndarray]:
    axes: list[np.ndarray] = []
    for index in range(len(polygon)):
        edge = polygon[(index + 1) % len(polygon)] - polygon[index]
        normal = np.asarray([-edge[1], edge[0]], dtype=float)
        norm = float(np.linalg.norm(normal))
        if norm > 1e-12:
            axes.append(normal / norm)
    return axes


def polygons_overlap(left: np.ndarray, right: np.ndarray, *, padding: float = 0.0) -> bool:
    if padding < 0:
        raise ValueError("padding must be non-negative")
    first = np.asarray(left, dtype=float)
    second = np.asarray(right, dtype=float)
    for axis in [*_polygon_axes(first), *_polygon_axes(second)]:
        left_min, left_max = _project_polygon(axis, first)
        right_min, right_max = _project_polygon(axis, second)
        if left_max + padding < right_min or right_max + padding < left_min:
            return False
    return True


def rect_polygon(rect: AxisAlignedRect) -> np.ndarray:
    min_x, max_x, min_y, max_y = rect.bounds
    return np.asarray(
        [[max_x, max_y], [max_x, min_y], [min_x, min_y], [min_x, max_y]],
        dtype=float,
    )


def _point_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    edge = end - start
    squared = float(edge @ edge)
    if squared <= 1e-12:
        return float(np.linalg.norm(point - start))
    t = float(np.clip(((point - start) @ edge) / squared, 0.0, 1.0))
    projection = start + t * edge
    return float(np.linalg.norm(point - projection))


def polygon_distance(left: np.ndarray, right: np.ndarray) -> float:
    first = np.asarray(left, dtype=float)
    second = np.asarray(right, dtype=float)
    if polygons_overlap(first, second):
        return 0.0
    best = math.inf
    for point in first:
        for index in range(len(second)):
            best = min(
                best,
                _point_segment_distance(
                    point,
                    second[index],
                    second[(index + 1) % len(second)],
                ),
            )
    for point in second:
        for index in range(len(first)):
            best = min(
                best,
                _point_segment_distance(
                    point,
                    first[index],
                    first[(index + 1) % len(first)],
                ),
            )
    return float(best)


def signed_slot_clearance(corners: np.ndarray, slot: ParkingSlot) -> float:
    slot_pose = Pose2D(slot.center_x, slot.center_y, slot.heading_deg)
    local = world_to_local(corners, slot_pose)
    half_length = slot.length / 2.0
    half_width = slot.width / 2.0
    longitudinal = half_length - np.abs(local[:, 0])
    lateral = half_width - np.abs(local[:, 1])
    return float(np.min(np.concatenate([longitudinal, lateral])))


def inside_slot_fraction(corners: np.ndarray, slot: ParkingSlot) -> float:
    polygon = slot_corners(slot)
    return float(sum(point_in_convex_polygon(point, polygon) for point in corners) / len(corners))


def obstacle_clearance(corners: np.ndarray, obstacles: Iterable[AxisAlignedRect]) -> float:
    distances = [polygon_distance(corners, rect_polygon(obstacle)) for obstacle in obstacles]
    return min(distances, default=math.inf)


def world_boundary_clearance(corners: np.ndarray, world_size: float) -> float:
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    distances = world_size - np.abs(corners)
    return float(np.min(distances))


def parking_quality(
    *,
    pose: Pose2D,
    footprint: VehicleFootprint,
    slot: ParkingSlot,
    obstacles: Iterable[AxisAlignedRect] = (),
    world_size: float = 32.0,
) -> ParkingQuality:
    corners = vehicle_corners(pose, footprint)
    obstacle_list = list(obstacles)
    slot_clearance = signed_slot_clearance(corners, slot)
    obstacle_gap = obstacle_clearance(corners, obstacle_list)
    world_gap = world_boundary_clearance(corners, world_size)
    collision = world_gap <= 0 or any(
        polygons_overlap(corners, rect_polygon(obstacle)) for obstacle in obstacle_list
    )
    center_distance = math.hypot(pose.x - slot.center_x, pose.y - slot.center_y)
    heading_error = abs(wrap_angle_deg(slot.heading_deg - pose.heading_deg))
    fraction = inside_slot_fraction(corners, slot)
    return ParkingQuality(
        center_distance=center_distance,
        heading_error_deg=heading_error,
        inside_fraction=fraction,
        min_slot_clearance=slot_clearance,
        obstacle_clearance=obstacle_gap,
        world_clearance=world_gap,
        fully_inside_slot=slot_clearance >= 0,
        collision=collision,
    )
