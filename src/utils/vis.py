# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>
"""Visualization utils for Argoverse MF scenarios."""

import math
from pathlib import Path
from typing import Final, Optional, Sequence, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
from av2.datasets.motion_forecasting.data_schema import (ArgoverseScenario,
                                                         ObjectType,
                                                         TrackCategory)
from av2.map.map_api import ArgoverseStaticMap
from av2.utils.typing import NDArrayFloat, NDArrayInt
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
from matplotlib.legend_handler import HandlerLineCollection
from matplotlib.patches import Rectangle

_PlotBounds = Tuple[float, float, float, float]

# Configure constants
_OBS_DURATION_TIMESTEPS: Final[int] = 50
_PRED_DURATION_TIMESTEPS: Final[int] = 60

_ESTIMATED_VEHICLE_LENGTH_M: Final[float] = 5
_ESTIMATED_VEHICLE_WIDTH_M: Final[float] = 2.5
_ESTIMATED_CYCLIST_LENGTH_M: Final[float] = 2.0
_ESTIMATED_CYCLIST_WIDTH_M: Final[float] = 0.7
_PLOT_BOUNDS_BUFFER_W: Final[float] = 80
_PLOT_BOUNDS_BUFFER_H: Final[float] = 80

_DRIVABLE_AREA_COLOR: Final[str] = "#7A7A7A"
_LANE_SEGMENT_COLOR: Final[str] = "#E0E0E0"
# _LANE_SEGMENT_COLOR: Final[str] = "#"

_DEFAULT_ACTOR_COLOR: Final[str] = "#815847"  # "#D3E8EF"
_HISTORY_COLOR: Final[str] = "#d34836"
_FOCAL_AGENT_COLOR: Final[str] = "#ff9a3a"
_AV_COLOR: Final[str] = "#007672"
_BOUNDING_BOX_ZORDER: Final[int] = 100

_STATIC_OBJECT_TYPES: Set[ObjectType] = {
    ObjectType.STATIC,
    ObjectType.BACKGROUND,
    ObjectType.CONSTRUCTION,
    ObjectType.RIDERLESS_BICYCLE,
}

def visualize_scenario(
    scenario: ArgoverseScenario,
    scenario_static_map: ArgoverseStaticMap,
    prediction: np.ndarray = None,
    timestep: int = 50,
    save_path: Path = None,
    title: str = "",
    tight=False,
    create_fig=True,
    show_future=True,
    show_history=True,
    show_map=True,
    best_pred=-1
) -> None:
    if create_fig:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    else:
        ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    if title != "": plt.title(title)

    # Plot static map elements and actor tracks
    if show_map: _plot_static_map_elements(scenario_static_map, True)
    cur_plot_bounds = _plot_actor_tracks(ax, scenario, timestep, show_history, show_future, show_map)
    plot_bounds = cur_plot_bounds

    if prediction is not None:
        print(f"Prediction shape: {prediction.shape}")
        
        # Debug: let's see what attributes the scenario actually has
        print(f"Scenario attributes: {[attr for attr in dir(scenario) if not attr.startswith('_')]}")
        
        # Si prediction tiene shape (A, M, T, 2) - múltiples agentes
        if len(prediction.shape) == 4:
            num_agents = prediction.shape[0]
            num_modes = prediction.shape[1]
            
            # Try different ways to access tracks in AV2
            tracks = None
            if hasattr(scenario, 'tracks'):
                if isinstance(scenario.tracks, dict):
                    tracks = list(scenario.tracks.values())
                elif isinstance(scenario.tracks, list):
                    tracks = scenario.tracks
            elif hasattr(scenario, 'tracked_objects'):
                tracks = scenario.tracked_objects
            else:
                print("Error: Cannot find tracks in scenario object")
                print(f"Available attributes: {[attr for attr in dir(scenario) if not attr.startswith('_')]}")
                return
            
            # Colores diferentes para cada agente
            agent_colors = [
                "#ffc187", "#87ceeb", "#98fb98", "#dda0dd", "#f0e68c", 
                "#ffa07a", "#20b2aa", "#87cefa", "#deb887", "#5f9ea0"
            ]
            
            # Debug coordinates
            print("=== DEBUGGING COORDINATES ===")
            print(f"Found {len(tracks)} tracks")
            
            for agent_idx in range(min(num_agents, len(tracks))):
                agent_pred = prediction[agent_idx].copy()  # (M, T, 2)
                color = agent_colors[agent_idx % len(agent_colors)]
                
                # Get agent's current position
                track = tracks[agent_idx]
                
                # Try different ways to access object states
                object_states = None
                if hasattr(track, 'object_states'):
                    object_states = track.object_states
                elif hasattr(track, 'states'):
                    object_states = track.states
                else:
                    print(f"Warning: Cannot find object states for track {agent_idx}")
                    continue
                
                # Check if timestep exists
                if timestep >= len(object_states):
                    print(f"Warning: Timestep {timestep} not available for agent {agent_idx}")
                    continue
                
                # Get position
                state = object_states[timestep]
                if hasattr(state, 'position'):
                    current_pos = np.array([state.position[0], state.position[1]])
                elif hasattr(state, 'pose'):
                    current_pos = np.array([state.pose.position[0], state.pose.position[1]])
                else:
                    print(f"Warning: Cannot find position for agent {agent_idx}")
                    continue
                
                # Check if prediction needs alignment
                pred_start = agent_pred[0, 0, :]  # First mode, first timestep
                distance_from_current = np.linalg.norm(pred_start - current_pos)
                
                if agent_idx < 3:  # Debug first 3 agents
                    track_id = getattr(track, 'track_id', f'unknown_{agent_idx}')
                    print(f"Agent {agent_idx} (track_id: {track_id}):")
                    print(f"  Current position: {current_pos}")
                    print(f"  Prediction starts at: {pred_start}")
                    print(f"  Distance: {distance_from_current:.2f}")
                
                # If prediction starts too far from current position, align it
                if distance_from_current > 10:  # Threshold in meters
                    print(f"  -> Aligning agent {agent_idx} predictions")
                    # Assume predictions are relative to origin, translate to current position
                    agent_pred = agent_pred + current_pos.reshape(1, 1, 2)
                
                # Now plot the aligned predictions
                if best_pred < 0:
                    # Mostrar todos los modos para este agente
                    _scatter_polylines(
                        agent_pred,  # (M, T, 2)
                        ax,
                        color=color,
                        grad_color=False,
                        alpha=0.6,
                        linewidth=2,
                        zorder=1000 + agent_idx,
                        arrow=False
                    )
                    # Puntos finales
                    plt.scatter(
                        agent_pred[:, -1, 0],
                        agent_pred[:, -1, 1],
                        color=color,
                        alpha=0.8,
                        zorder=2000 + agent_idx,
                        marker="*",
                        s=150
                    )
                    
                else:
                    # Mostrar todos los modos con baja opacidad
                    _scatter_polylines(
                        agent_pred,
                        ax,
                        color=color,
                        grad_color=False,
                        alpha=0.1,
                        linewidth=2,
                        zorder=1000 + agent_idx,
                        arrow=False
                    )
                    # Mostrar solo el mejor modo con alta opacidad
                    if best_pred < num_modes:
                        best_prediction = agent_pred[best_pred:best_pred+1]  # (1, T, 2)
                        _scatter_polylines(
                            best_prediction,
                            ax,
                            color=color,
                            grad_color=False,
                            alpha=1.0,
                            linewidth=3,
                            zorder=1000 + agent_idx,
                            arrow=False
                        )
                        plt.scatter(
                            best_prediction[0, -1, 0],
                            best_prediction[0, -1, 1],
                            color=color,
                            alpha=1.0,
                            zorder=2000 + agent_idx,
                            marker="*",
                            s=200
                        )
            
            print("=" * 30)
        
        # Si prediction tiene shape (M, T, 2) - un solo agente (compatibilidad con versión anterior)
        elif len(prediction.shape) == 3:
            # Get the focal agent's current position for alignment
            focal_track = None
            
            # Try to find focal track
            if hasattr(scenario, 'focal_track_id'):
                focal_track_id = scenario.focal_track_id
                
                # Find the focal track
                tracks = None
                if hasattr(scenario, 'tracks'):
                    if isinstance(scenario.tracks, dict):
                        focal_track = scenario.tracks.get(focal_track_id)
                    elif isinstance(scenario.tracks, list):
                        for track in scenario.tracks:
                            if getattr(track, 'track_id', None) == focal_track_id:
                                focal_track = track
                                break
            
            # If no focal track found, use first available track
            if focal_track is None:
                print("Warning: Focal track not found, using first available track")
                if hasattr(scenario, 'tracks'):
                    if isinstance(scenario.tracks, dict):
                        focal_track = list(scenario.tracks.values())[0]
                    elif isinstance(scenario.tracks, list):
                        focal_track = scenario.tracks[0]
                        
            if focal_track is None:
                print("Error: No tracks found in scenario")
                return
                
            # Get object states
            object_states = None
            if hasattr(focal_track, 'object_states'):
                object_states = focal_track.object_states
            elif hasattr(focal_track, 'states'):
                object_states = focal_track.states
                
            if object_states is None or timestep >= len(object_states):
                print(f"Warning: Cannot access timestep {timestep} for focal agent")
                return
                
            # Get position
            state = object_states[timestep]
            if hasattr(state, 'position'):
                current_pos = np.array([state.position[0], state.position[1]])
            elif hasattr(state, 'pose'):
                current_pos = np.array([state.pose.position[0], state.pose.position[1]])
            else:
                print("Warning: Cannot find position for focal agent")
                return
            
            # Check if prediction needs alignment
            pred_start = prediction[0, 0, :]  # First mode, first timestep
            distance_from_current = np.linalg.norm(pred_start - current_pos)
            
            aligned_prediction = prediction.copy()
            if distance_from_current > 10:  # Threshold in meters
                print(f"Aligning single agent prediction (distance: {distance_from_current:.2f}m)")
                aligned_prediction = prediction + current_pos.reshape(1, 1, 2)
            
            if best_pred < 0:
                _scatter_polylines(
                    aligned_prediction[:, :, :],
                    ax,
                    color="#ffc187",
                    grad_color=False,
                    alpha=0.8,
                    linewidth=3,
                    zorder=1000,
                    arrow=False
                )
                plt.scatter(
                    aligned_prediction[:, -1, 0],
                    aligned_prediction[:, -1, 1],
                    color="#ff993b",
                    alpha=1,
                    zorder=2000,
                    marker="*",
                    s=200
                )
            else:
                _scatter_polylines(
                    aligned_prediction[:, :, :],
                    ax,
                    color="#ffc187",
                    grad_color=False,
                    alpha=0.1,
                    linewidth=3,
                    zorder=1000,
                    arrow=False
                )
                plt.scatter(
                    aligned_prediction[:, -1, 0],
                    aligned_prediction[:, -1, 1],
                    color="#ff993b",
                    alpha=0.3,
                    zorder=2000,
                    marker="*",
                    s=200
                )
                if best_pred < aligned_prediction.shape[0]:
                    best_prediction = aligned_prediction[best_pred:best_pred+1, :, :]
                    _scatter_polylines(
                        best_prediction,
                        ax,
                        color="#ffc187",
                        grad_color=False,
                        alpha=1.0,
                        linewidth=4,
                        zorder=1000,
                        arrow=False
                    )
                    plt.scatter(
                        best_prediction[0, -1, 0],
                        best_prediction[0, -1, 1],
                        color="#ff993b",
                        alpha=1.0,
                        zorder=2000,
                        marker="*",
                        s=200
                    )

    plt.axis("equal")
    plt.xlim(
        plot_bounds[0] - _PLOT_BOUNDS_BUFFER_W, plot_bounds[0] + _PLOT_BOUNDS_BUFFER_W
    )
    plt.ylim(
        plot_bounds[1] - _PLOT_BOUNDS_BUFFER_H, plot_bounds[1] + _PLOT_BOUNDS_BUFFER_H
    )
    if tight: plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, pad_inches=0)
        plt.close()
    if create_fig: return fig

def _plot_static_map_elements(
    static_map: ArgoverseStaticMap, show_ped_xings: bool = False
) -> None:
    """Plot all static map elements associated with an Argoverse scenario.

    Args:
        static_map: Static map containing elements to be plotted.
        show_ped_xings: Configures whether pedestrian crossings should be plotted.
    """
    # Plot drivable areas
    _plot_polygons([drivable_area.xyz for drivable_area in static_map.vector_drivable_areas.values()], alpha=0.3, color=_DRIVABLE_AREA_COLOR) 
 
    # Plot lane segments
    for lane_segment in static_map.vector_lane_segments.values():
        centerline = static_map.get_lane_segment_centerline(lane_segment.id)
        _plot_polylines(
            [centerline], line_width=2.0, color="#000000", alpha=0.2, style="--"
        )
        _plot_polylines(
            [centerline],
            line_width=3,
            color=[0.3, 0.3, 0.3],
            endpoint=True,
            zorder=98,
        )

    # Plot pedestrian crossings
    if show_ped_xings:
        for ped_xing in static_map.vector_pedestrian_crossings.values():
            _plot_polylines(
                [ped_xing.edge1.xyz, ped_xing.edge2.xyz],
                alpha=1.0,
                color="white",
            )


def _plot_actor_tracks(
    ax: plt.Axes,
    scenario: ArgoverseScenario,
    timestep: int,
    show_history: bool,
    show_future: bool,
    show_map: bool
) -> Optional[_PlotBounds]:
    """Plot all actor tracks (up to a particular time step) associated with an Argoverse scenario.

    Args:
        ax: Axes on which actor tracks should be plotted.
        scenario: Argoverse scenario for which to plot actor tracks.
        timestep: Tracks are plotted for all actor data up to the specified time step.

    Returns:
        track_bounds: (x_min, x_max, y_min, y_max) bounds for the extent of actor tracks.
    """
    track_bounds = None
    focal_id = scenario.focal_track_id

    for track in scenario.tracks:
        if track.track_id != focal_id and not show_map: continue
        # Get timesteps for which actor data is valid
        actor_timesteps: NDArrayInt = np.array(
            [
                object_state.timestep
                for object_state in track.object_states
                if object_state.timestep <= timestep
            ]
        )

        if actor_timesteps.shape[0] < 1 or actor_timesteps[-1] != timestep:
            continue

        future_trajectory: NDArrayFloat = np.array(
            [
                list(object_state.position)
                for object_state in track.object_states
                if object_state.timestep > timestep
            ]
        )
        # Get actor trajectory and heading history
        history_trajectory: NDArrayFloat = np.array(
            [
                list(object_state.position)
                for object_state in track.object_states
                if object_state.timestep <= timestep
            ]
        )
        actor_headings: NDArrayFloat = np.array(
            [
                object_state.heading
                for object_state in track.object_states
                if object_state.timestep <= timestep
            ]
        )

        # Plot polyline for focal agent location history
        track_color = _DEFAULT_ACTOR_COLOR
        from matplotlib.colors import LinearSegmentedColormap

        predcmp = LinearSegmentedColormap.from_list("pred", [[105/255.0, 172/255.0, 160/255.0], [180/255.0, 214/255.0, 208/255.0]], N=256)
        if track.category == TrackCategory.FOCAL_TRACK and len(future_trajectory) > 0 and show_future:
            _scatter_polylines(
                [future_trajectory],
                cmap=predcmp,#"Greens",
                linewidth=12,
                reverse=False,
                arrow=False,
                alpha=1.0,
                zorder=199
            )
        elif track.object_type in _STATIC_OBJECT_TYPES:
            continue

        track_color = _DEFAULT_ACTOR_COLOR
        hist_cmap = LinearSegmentedColormap.from_list("pred", [[126/255.0, 135/255.0, 167/255.0], [44/255.0, 51/255.0, 80/255.0]], N=128) 
        if show_history: _scatter_polylines([history_trajectory], cmap=hist_cmap, linewidth=8, arrow=False, alpha=0.9)
        
        if track.track_id == focal_id:
            track_bounds = history_trajectory[-1]
            track_color = _FOCAL_AGENT_COLOR

        # Plot bounding boxes for all vehicles and cyclists
        if track.object_type == ObjectType.VEHICLE:
            _plot_actor_bounding_box(
                ax,
                history_trajectory[-1],
                actor_headings[-1],
                track_color,
                (_ESTIMATED_VEHICLE_LENGTH_M, _ESTIMATED_VEHICLE_WIDTH_M),
            )
        elif (
            track.object_type == ObjectType.CYCLIST
            or track.object_type == ObjectType.MOTORCYCLIST
        ):
            _plot_actor_bounding_box(
                ax,
                history_trajectory[-1],
                actor_headings[-1],
                track_color,
                (_ESTIMATED_CYCLIST_LENGTH_M, _ESTIMATED_CYCLIST_WIDTH_M),
            )
        else:
            plt.plot(
                history_trajectory[-1, 0],
                history_trajectory[-1, 1],
                "o",
                color=track_color,
                markersize=11,
                zorder=999
            )

    return track_bounds


class HandlerColorLineCollection(HandlerLineCollection):
    def __init__(
        self,
        reverse: bool = False,
        marker_pad: float = ...,
        numpoints: None = ...,
        **kwargs,
    ) -> None:
        super().__init__(marker_pad, numpoints, **kwargs)
        self.reverse = reverse

    def create_artists(
        self, legend, artist, xdescent, ydescent, width, height, fontsize, trans
    ):
        x = np.linspace(0, width, self.get_numpoints(legend) + 1)
        y = np.zeros(self.get_numpoints(legend) + 1) + height / 2.0 - ydescent
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap=artist.cmap, transform=trans)
        lc.set_array(x if not self.reverse else x[::-1])
        lc.set_linewidth(artist.get_linewidth())
        return [lc]


def _plot_polylines(
    polylines: Sequence[NDArrayFloat],
    *,
    style: str = "-",
    line_width: float = 1.0,
    alpha: float = 1.0,
    color: str = "r",
    endpoint: bool = False,
    **kwargs,
) -> None:
    """Plot a group of polylines with the specified config.

    Args:
        polylines: Collection of (N, 2) polylines to plot.
        style: Style of the line to plot (e.g. `-` for solid, `--` for dashed)
        line_width: Desired width for the plotted lines.
        alpha: Desired alpha for the plotted lines.
        color: Desired color for the plotted lines.
    """
    for polyline in polylines:
        plt.plot(
            polyline[:, 0],
            polyline[:, 1],
            style,
            linewidth=line_width,
            color=color,
            alpha=alpha,
            **kwargs,
        )
        if endpoint:
            plt.scatter(polyline[0, 0], polyline[0, 1], color=color, s=15, **kwargs)


def get_polyline_arc_length(xy: np.ndarray) -> np.ndarray:
    """Get the arc length of each point in a polyline"""
    diff = xy[1:] - xy[:-1]
    displacement = np.sqrt(diff[:, 0] ** 2 + diff[:, 1] ** 2)
    arc_length = np.cumsum(displacement)
    return np.concatenate((np.zeros(1), arc_length), axis=0)


def interpolate_lane(xy: np.ndarray, arc_length: np.ndarray, steps: np.ndarray):
    xy_inter = np.empty((steps.shape[0], 2), dtype=xy.dtype)
    xy_inter[:, 0] = np.interp(steps, xp=arc_length, fp=xy[:, 0])
    xy_inter[:, 1] = np.interp(steps, xp=arc_length, fp=xy[:, 1])
    return xy_inter


def interpolate_centerline(xy: np.ndarray, n_points: int):
    arc_length = get_polyline_arc_length(xy)
    steps = np.linspace(0, arc_length[-1], n_points)
    xy_inter = np.empty((steps.shape[0], 2), dtype=xy.dtype)
    xy_inter[:, 0] = np.interp(steps, xp=arc_length, fp=xy[:, 0])
    xy_inter[:, 1] = np.interp(steps, xp=arc_length, fp=xy[:, 1])
    return xy_inter


def _scatter_polylines(
    polylines: Sequence[NDArrayFloat],
    cmap="spring",
    linewidth=3,
    arrow: bool = True,
    reverse: bool = False,
    alpha=0.5,
    zorder=100,
    grad_color: bool = True,
    color=None,
    linestyle="-",
) -> None:
    """Plot a group of polylines with the specified config.

    Args:
        polylines: Collection of (N, 2) polylines to plot.
        style: Style of the line to plot (e.g. `-` for solid, `--` for dashed)
        line_width: Desired width for the plotted lines.
        alpha: Desired alpha for the plotted lines.
        color: Desired color for the plotted lines.
    """
    ax = plt.gca()
    for polyline in polylines:
        inter_poly = interpolate_centerline(polyline, 60)

        if arrow:
            point = inter_poly[-1]
            diff = inter_poly[-1] - inter_poly[-2]
            diff = diff / np.linalg.norm(diff)
            if grad_color:
                c = plt.cm.get_cmap(cmap)(0)
            else:
                c = color
            arrow = ax.quiver(
                point[0],
                point[1],
                diff[0],
                diff[1],
                alpha=alpha,
                scale_units="xy",
                scale=0.25,
                minlength=0.5,
                zorder=zorder - 1,
                color=c,
            )

        if grad_color:
            arc = get_polyline_arc_length(inter_poly)
            polyline = inter_poly.reshape(-1, 1, 2)
            segment = np.concatenate([polyline[:-1], polyline[1:]], axis=1)
            norm = plt.Normalize(arc.min(), arc.max())
            lc = LineCollection(
                segment, cmap=cmap, norm=norm, zorder=zorder, alpha=alpha
            )
            lc.set_array(arc if not reverse else arc[::-1])
            lc.set_linewidth(linewidth)
            ax.add_collection(lc)
        else:
            ax.plot(
                inter_poly[:, 0],
                inter_poly[:, 1],
                color=color,
                linewidth=linewidth,
                zorder=zorder,
                alpha=alpha,
                linestyle=linestyle,
            )


def _plot_polygons(
    polygons: Sequence[NDArrayFloat], *, alpha: float = 1.0, color: str = "r"
) -> None:
    """Plot a group of filled polygons with the specified config.

    Args:
        polygons: Collection of polygons specified by (N,2) arrays of vertices.
        alpha: Desired alpha for the polygon fill.
        color: Desired color for the polygon.
    """
    for polygon in polygons:
        plt.fill(
            polygon[:, 0],
            polygon[:, 1],
            fc=to_rgba(color, alpha),
            ec="black",
            linewidth=0,
            zorder=2,
        )


def _plot_actor_bounding_box(
    ax: plt.Axes,
    cur_location: NDArrayFloat,
    heading: float,
    color: str,
    bbox_size: Tuple[float, float],
) -> None:
    """Plot an actor bounding box centered on the actor's current location.

    Args:
        ax: Axes on which actor bounding box should be plotted.
        cur_location: Current location of the actor (2,).
        heading: Current heading of the actor (in radians).
        color: Desired color for the bounding box.
        bbox_size: Desired size for the bounding box (length, width).
    """
    (bbox_length, bbox_width) = bbox_size

    # Compute coordinate for pivot point of bounding box
    d = np.hypot(bbox_length, bbox_width)
    theta_2 = math.atan2(bbox_width, bbox_length)
    pivot_x = cur_location[0] - (d / 2) * math.cos(heading + theta_2)
    pivot_y = cur_location[1] - (d / 2) * math.sin(heading + theta_2)

    vehicle_bounding_box = Rectangle(
        (pivot_x, pivot_y),
        bbox_length,
        bbox_width,
        np.degrees(heading),
        zorder=_BOUNDING_BOX_ZORDER + 100,
        fc=color,
        ec="dimgrey",
        alpha=1.0,
    )
    ax.add_patch(vehicle_bounding_box)
