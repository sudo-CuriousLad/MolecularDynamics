"""A small interactive Pygame viewer for the 3D particle model in simulation.ipynb.

Run with: python -m pip install numpy pygame-ce
         python simulation_pygame.py

pygame-ce provides the standard ``pygame`` module and supports newer Python builds.
"""

from __future__ import annotations

import math

try:
    import pygame
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Pygame is required. Install the app dependencies with "
        "'python -m pip install numpy pygame-ce'."
    ) from exc

import numpy as np


WIDTH, HEIGHT = 1280, 820
MIN_WIDTH, MIN_HEIGHT = 1050, 700
FPS = 60
SIMULATION_HZ = 20

BOUNDS = 400.0
DT = 0.5
INTEGRATION_STEPS = 50
INTEGRATION_DT = DT / INTEGRATION_STEPS
PARTICLE_MASS = 1.0
INITIAL_PARTICLES = 20
INITIAL_SPEED = 10.0
INITIAL_K = 5.0
INITIAL_RADIUS = 100.0
MIN_PARTICLES, MAX_PARTICLES = 2, 80
MIN_K, MAX_K = 1.0, 20.0
MIN_RADIUS, MAX_RADIUS = 100.0, 200.0
HISTORY_SECONDS = 5.0

BACKGROUND = (245, 247, 250)
CARD = (255, 255, 255)
TEXT = (35, 42, 52)
SECONDARY_TEXT = (112, 121, 134)
HAIRLINE = (229, 233, 239)
ACCENT = (42, 111, 219)
ACCENT_DARK = (30, 86, 178)
ACCENT_LIGHT = (224, 237, 255)
CORAL = (242, 132, 112)
CHART = (42, 111, 219)
KINETIC_COLOR = (48, 151, 111)
BOX_EDGE = (177, 190, 207)


def sample_spherical(npoints: int, ndim: int = 3) -> np.ndarray:
    vectors = np.random.randn(ndim, npoints)
    norms = np.linalg.norm(vectors, axis=0)
    norms[norms == 0] = 1.0
    return vectors / norms


def make_atoms(n_count: int) -> np.ndarray:
    """Create the notebook's [position, velocity, cell] particle array."""
    atoms = np.zeros((n_count, 9), dtype=np.float64)
    atoms[:, :3] = np.floor(sample_spherical(n_count).T * 100.0)
    atoms[:, 3:6] = sample_spherical(n_count).T * INITIAL_SPEED
    update_cells(atoms, INITIAL_RADIUS)
    return atoms


def update_cells(atoms: np.ndarray, interaction_radius: float) -> None:
    atoms[:, 6:9] = np.floor(atoms[:, :3] / interaction_radius)


def get_neighbours(
    index: int,
    atoms: np.ndarray,
    cells: dict[tuple[int, int, int], list[int]],
    interaction_radius: float,
) -> np.ndarray:
    """Find particles within range using the surrounding 3x3x3 cell block."""
    cell = atoms[index, 6:9].astype(int)
    candidates: list[int] = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                candidates.extend(
                    cells.get((cell[0] + dx, cell[1] + dy, cell[2] + dz), ())
                )

    candidates = [candidate for candidate in candidates if candidate != index]
    if not candidates:
        return np.empty((0, 9), dtype=atoms.dtype)

    neighbours = atoms[candidates]
    distances = np.linalg.norm(neighbours[:, :3] - atoms[index, :3], axis=1)
    return neighbours[(distances > 0.0) & (distances < interaction_radius)]


def force_func(
    distance: np.ndarray | float, stiffness: float, interaction_radius: float
) -> np.ndarray:
    return -stiffness * (distance - interaction_radius)


def calculate_forces(
    atoms: np.ndarray, stiffness: float, interaction_radius: float
) -> np.ndarray:
    update_cells(atoms, interaction_radius)
    displacement = atoms[:, None, :3] - atoms[None, :, :3]
    distances = np.linalg.norm(displacement, axis=2)
    interacting = (distances > 0.0) & (distances < interaction_radius)
    magnitudes = np.zeros_like(distances)
    magnitudes[interacting] = force_func(
        distances[interacting], stiffness, interaction_radius
    )
    directions = np.divide(
        displacement,
        distances[:, :, None],
        out=np.zeros_like(displacement),
        where=distances[:, :, None] > 0.0,
    )
    return np.sum(magnitudes[:, :, None] * directions, axis=1)


def step(
    atoms: np.ndarray,
    stiffness: float,
    interaction_radius: float,
    bounds: float = BOUNDS,
    dt: float = INTEGRATION_DT,
) -> np.ndarray:
    """Advance one small step with velocity Verlet and elastic wall reflections."""
    forces = calculate_forces(atoms, stiffness, interaction_radius)
    half_velocities = atoms[:, 3:6] + 0.5 * (forces / PARTICLE_MASS) * dt
    positions = atoms[:, :3] + half_velocities * dt

    shifted = positions + bounds
    period = 4.0 * bounds
    folded = np.remainder(shifted, period)
    positions[:] = np.where(folded <= 2.0 * bounds, folded - bounds, 3.0 * bounds - folded)
    wall_segments = np.floor(shifted / (2.0 * bounds)).astype(np.int64)
    reflected = np.remainder(wall_segments, 2) == 1
    half_velocities[reflected] *= -1.0

    atoms[:, :3] = positions
    atoms[:, 3:6] = half_velocities
    forces = calculate_forces(atoms, stiffness, interaction_radius)
    atoms[:, 3:6] += 0.5 * (forces / PARTICLE_MASS) * dt
    return atoms


def kinetic_energy(atoms: np.ndarray) -> float:
    return float(0.5 * PARTICLE_MASS * np.sum(atoms[:, 3:6] ** 2))


def potential_energy(
    atoms: np.ndarray, stiffness: float, interaction_radius: float
) -> float:
    """Return each active pair spring's potential energy exactly once."""
    displacement = atoms[:, None, :3] - atoms[None, :, :3]
    distances = np.linalg.norm(displacement, axis=2)
    upper_triangle = np.triu(np.ones(distances.shape, dtype=bool), k=1)
    interacting_pairs = upper_triangle & (distances > 0.0) & (
        distances < interaction_radius
    )
    extensions = distances[interacting_pairs] - interaction_radius
    return float(0.5 * stiffness * np.sum(extensions**2))


def energy_components(
    atoms: np.ndarray, stiffness: float, interaction_radius: float
) -> tuple[float, float, float]:
    kinetic = kinetic_energy(atoms)
    potential = potential_energy(atoms, stiffness, interaction_radius)
    return kinetic, potential, kinetic + potential


def mix_color(start: tuple[int, int, int], end: tuple[int, int, int], t: float):
    t = min(1.0, max(0.0, t))
    return tuple(round(a + (b - a) * t) for a, b in zip(start, end))


class ParticleSimulation:
    def __init__(self) -> None:
        self.particle_count = INITIAL_PARTICLES
        self.stiffness = INITIAL_K
        self.interaction_radius = INITIAL_RADIUS
        self.paused = False
        self.simulated_time = 0.0
        self.reset()

    def reset(self) -> None:
        self.atoms = make_atoms(self.particle_count)
        self.simulated_time = 0.0
        self.times = [0.0]
        self.kinetic_energies = []
        self.potential_energies = []
        self.total_energies = []
        self.reset_energy_history()

    def reset_energy_history(self) -> None:
        kinetic, potential, total = energy_components(
            self.atoms, self.stiffness, self.interaction_radius
        )
        self.simulated_time = 0.0
        self.reference_energy = total
        self.times = [0.0]
        self.kinetic_energies = [kinetic]
        self.potential_energies = [potential]
        self.total_energies = [total]

    def advance_one_step(self) -> None:
        step(
            self.atoms,
            self.stiffness,
            self.interaction_radius,
            dt=INTEGRATION_DT,
        )

    def set_particle_count(self, count: int) -> None:
        self.particle_count = min(MAX_PARTICLES, max(MIN_PARTICLES, count))
        self.reset()

    def advance(self) -> None:
        if self.paused:
            return
        for _ in range(INTEGRATION_STEPS):
            self.advance_one_step()
        self.simulated_time += DT
        self.times.append(self.simulated_time)
        kinetic, potential, total = energy_components(
            self.atoms, self.stiffness, self.interaction_radius
        )
        self.kinetic_energies.append(kinetic)
        self.potential_energies.append(potential)
        self.total_energies.append(total)
        cutoff = self.simulated_time - HISTORY_SECONDS
        while len(self.times) > 2 and self.times[1] < cutoff:
            self.times.pop(0)
            self.kinetic_energies.pop(0)
            self.potential_energies.pop(0)
            self.total_energies.pop(0)


def font(size: int, weight: str = "regular") -> pygame.font.Font:
    candidates = (
        ("SF Pro Display", "Avenir Next", "Helvetica Neue", "Arial")
        if weight == "bold"
        else ("SF Pro Text", "Avenir Next", "Helvetica Neue", "Arial")
    )
    for name in candidates:
        result = pygame.font.match_font(name, bold=weight == "bold")
        if result:
            return pygame.font.Font(result, size)
    return pygame.font.Font(None, size)


class PygameApp:
    def __init__(self) -> None:
        pygame.init()
        pygame.display.set_caption("Molecular Motion")
        self.screen = pygame.display.set_mode(
            (WIDTH, HEIGHT), pygame.RESIZABLE | pygame.SCALED
        )
        self.clock = pygame.time.Clock()
        self.title_font = font(27, "bold")
        self.heading_font = font(16, "bold")
        self.body_font = font(14)
        self.small_font = font(12)
        self.value_font = font(18, "bold")
        self.simulation = ParticleSimulation()
        self.yaw = -0.66
        self.pitch = 0.56
        self.dragging_view = False
        self.dragging_slider: str | None = None
        self.slider_rects: dict[str, pygame.Rect] = {}
        self.buttons: dict[str, pygame.Rect] = {}

    def text(
        self,
        surface: pygame.Surface,
        value: str,
        position: tuple[int, int],
        color: tuple[int, int, int] = TEXT,
        selected_font: pygame.font.Font | None = None,
    ) -> pygame.Rect:
        image = (selected_font or self.body_font).render(value, True, color)
        return surface.blit(image, position)

    def card(self, rect: pygame.Rect, radius: int = 18) -> None:
        pygame.draw.rect(self.screen, CARD, rect, border_radius=radius)
        pygame.draw.rect(self.screen, HAIRLINE, rect, width=1, border_radius=radius)

    def layout(self) -> tuple[pygame.Rect, pygame.Rect, pygame.Rect]:
        width, height = self.screen.get_size()
        margin, gap = 26, 20
        control = pygame.Rect(margin, height - 148, width - margin * 2, 122)
        content_y = 94
        content_h = control.top - content_y - 18
        side_w = min(360, max(310, int(width * 0.29)))
        main_w = width - margin * 2 - gap - side_w
        main = pygame.Rect(margin, content_y, main_w, content_h)
        energy = pygame.Rect(main.right + gap, content_y, side_w, content_h)
        return main, energy, control

    def draw_header(self) -> None:
        width, _ = self.screen.get_size()
        self.text(self.screen, "Molecular Motion", (28, 22), TEXT, self.title_font)
        self.text(
            self.screen,
            "A small 3D particle simulation",
            (30, 56),
            SECONDARY_TEXT,
            self.body_font,
        )
        status_text = "PAUSED" if self.simulation.paused else "RUNNING"
        status_color = SECONDARY_TEXT if self.simulation.paused else ACCENT_DARK
        status = self.heading_font.render(status_text, True, status_color)
        dot_x = width - status.get_width() - 52
        dot_y = 36
        pygame.draw.circle(
            self.screen,
            SECONDARY_TEXT if self.simulation.paused else (47, 166, 112),
            (dot_x, dot_y + 7),
            4,
        )
        self.screen.blit(status, (dot_x + 14, dot_y))
        self.text(
            self.screen,
            "DRAG TO ORBIT",
            (width - 164, 60),
            SECONDARY_TEXT,
            self.small_font,
        )

    def project(
        self, position: np.ndarray, center: tuple[float, float], scale: float
    ) -> tuple[float, float, float]:
        x, y, z = position
        cos_yaw, sin_yaw = math.cos(self.yaw), math.sin(self.yaw)
        rotated_x = cos_yaw * x - sin_yaw * y
        rotated_y = sin_yaw * x + cos_yaw * y
        cos_pitch, sin_pitch = math.cos(self.pitch), math.sin(self.pitch)
        projected_y = cos_pitch * rotated_y - sin_pitch * z
        depth = sin_pitch * rotated_y + cos_pitch * z
        return (
            center[0] + rotated_x * scale,
            center[1] - projected_y * scale,
            depth,
        )

    def draw_simulation(self, rect: pygame.Rect) -> None:
        self.card(rect)
        self.text(self.screen, "Particle field", (rect.x + 22, rect.y + 19), TEXT, self.heading_font)
        self.text(
            self.screen,
            "Short-range interactions in a reflective 3D box",
            (rect.x + 22, rect.y + 44),
            SECONDARY_TEXT,
            self.small_font,
        )
        self.text(
            self.screen,
            f"{self.simulation.particle_count} particles",
            (rect.right - 120, rect.y + 23),
            ACCENT_DARK,
            self.small_font,
        )

        plot = pygame.Rect(rect.x + 18, rect.y + 75, rect.width - 36, rect.height - 98)
        center = (plot.centerx, plot.centery + 2)
        scale = min((plot.width - 70) / (4 * BOUNDS), (plot.height - 45) / (4 * BOUNDS))
        corners = [
            np.array([x * BOUNDS, y * BOUNDS, z * BOUNDS], dtype=float)
            for x in (-1, 1)
            for y in (-1, 1)
            for z in (-1, 1)
        ]
        projected = [self.project(corner, center, scale) for corner in corners]

        # Connect corners that differ along exactly one axis.
        for i, point in enumerate(projected):
            for j in range(i + 1, len(projected)):
                if sum(a != b for a, b in zip(corners[i], corners[j])) == 1:
                    pygame.draw.line(
                        self.screen,
                        BOX_EDGE,
                        (round(point[0]), round(point[1])),
                        (round(projected[j][0]), round(projected[j][1])),
                        1,
                    )

        speeds = np.linalg.norm(self.simulation.atoms[:, 3:6], axis=1)
        max_speed = max(float(np.max(speeds)), 1.0)
        particles = []
        for atom, speed in zip(self.simulation.atoms, speeds):
            point = self.project(atom[:3], center, scale)
            particles.append((point[2], point, speed))
        particles.sort(key=lambda item: item[0])

        for _, point, speed in particles:
            x, y, _ = point
            speed_fraction = min(float(speed) / max_speed, 1.0)
            radius = 4 + round(speed_fraction * 2)
            color = mix_color(ACCENT, CORAL, speed_fraction * 0.45)
            pygame.draw.circle(self.screen, mix_color(color, CARD, 0.72), (round(x), round(y)), radius + 3)
            pygame.draw.circle(self.screen, color, (round(x), round(y)), radius)
            pygame.draw.circle(
                self.screen,
                (255, 255, 255),
                (round(x - 1), round(y - 1)),
                max(1, radius // 3),
            )

        self.text(self.screen, "x", (plot.right - 20, plot.centery), SECONDARY_TEXT, self.small_font)
        self.text(self.screen, "y", (plot.centerx - 6, plot.y + 4), SECONDARY_TEXT, self.small_font)
        self.text(self.screen, "z", (plot.centerx + 8, plot.bottom - 20), SECONDARY_TEXT, self.small_font)

    def draw_energy(self, rect: pygame.Rect) -> None:
        self.card(rect)
        self.text(self.screen, "Energy conservation", (rect.x + 22, rect.y + 19), TEXT, self.heading_font)
        kinetic = self.simulation.kinetic_energies[-1]
        potential = self.simulation.potential_energies[-1]
        total = self.simulation.total_energies[-1]
        drift = total - self.simulation.reference_energy
        drift_percent = 100.0 * drift / max(abs(self.simulation.reference_energy), 1e-12)
        drift_color = (47, 135, 91) if abs(drift_percent) < 0.5 else (190, 119, 43)
        self.text(
            self.screen,
            f"{total:,.2f}",
            (rect.x + 22, rect.y + 49),
            TEXT,
            self.value_font,
        )
        self.text(
            self.screen,
            f"ΔE {drift:+,.2f}  ({drift_percent:+.2f}%)",
            (rect.right - 166, rect.y + 54),
            drift_color,
            self.small_font,
        )
        self.text(
            self.screen,
            f"Kinetic  {kinetic:,.2f}",
            (rect.x + 22, rect.y + 79),
            KINETIC_COLOR,
            self.small_font,
        )
        self.text(
            self.screen,
            f"Potential  {potential:,.2f}",
            (rect.x + 182, rect.y + 79),
            CORAL,
            self.small_font,
        )

        graph = pygame.Rect(rect.x + 22, rect.y + 143, rect.width - 44, rect.height - 195)
        if graph.width < 20 or graph.height < 20:
            return
        pygame.draw.line(self.screen, HAIRLINE, (graph.left, graph.bottom), (graph.right, graph.bottom), 1)
        pygame.draw.line(self.screen, HAIRLINE, (graph.left, graph.top), (graph.left, graph.bottom), 1)

        window_start = max(0.0, self.simulation.simulated_time - HISTORY_SECONDS)
        visible_indices = [
            index
            for index, time in enumerate(self.simulation.times)
            if time >= window_start
        ]
        if len(visible_indices) > 1:
            visible_times = [self.simulation.times[i] for i in visible_indices]
            series = (
                (self.simulation.total_energies, CHART),
                (self.simulation.kinetic_energies, KINETIC_COLOR),
                (self.simulation.potential_energies, CORAL),
            )
            values = [energies[i] for energies, _ in series for i in visible_indices]
            minimum, maximum = min(values), max(values)
            padding = max(1.0, (maximum - minimum) * 0.12)
            low, high = max(0.0, minimum - padding), maximum + padding
            if high <= low:
                high = low + 1.0
            time_span = max(HISTORY_SECONDS, self.simulation.simulated_time) - window_start
            time_span = max(time_span, DT)
            for energies, color in series:
                points = []
                for index, time in zip(visible_indices, visible_times):
                    x = graph.left + (time - window_start) / time_span * graph.width
                    y = graph.bottom - (energies[index] - low) / (high - low) * graph.height
                    points.append((round(x), round(y)))
                pygame.draw.lines(self.screen, color, False, points, 3)
                pygame.draw.circle(self.screen, color, points[-1], 4)

        self.text(
            self.screen,
            f"{max(0.0, self.simulation.simulated_time - HISTORY_SECONDS):.1f}s",
            (graph.left, graph.bottom + 10),
            SECONDARY_TEXT,
            self.small_font,
        )
        self.text(
            self.screen,
            f"{self.simulation.simulated_time:.1f}s",
            (graph.right - 36, graph.bottom + 10),
            SECONDARY_TEXT,
            self.small_font,
        )
        self.text(
            self.screen,
            f"Elapsed  {self.simulation.simulated_time:.1f} s",
            (rect.x + 22, rect.bottom - 33),
            SECONDARY_TEXT,
            self.small_font,
        )
        legend_y = rect.y + 116
        for x, color, label in (
            (rect.x + 22, CHART, "Total"),
            (rect.x + 102, KINETIC_COLOR, "Kinetic"),
            (rect.x + 206, CORAL, "Potential"),
        ):
            pygame.draw.circle(self.screen, color, (x + 4, legend_y + 6), 4)
            self.text(
                self.screen,
                label,
                (x + 13, legend_y),
                SECONDARY_TEXT,
                self.small_font,
            )

    def draw_slider(
        self,
        name: str,
        label: str,
        value: float,
        minimum: float,
        maximum: float,
        rect: pygame.Rect,
        value_text: str,
    ) -> None:
        self.slider_rects[name] = rect
        self.text(self.screen, label, (rect.x, rect.y - 22), SECONDARY_TEXT, self.small_font)
        shown = self.body_font.render(value_text, True, TEXT)
        self.screen.blit(shown, (rect.right - shown.get_width(), rect.y - 24))
        pygame.draw.line(self.screen, HAIRLINE, rect.midleft, rect.midright, 4)
        fraction = (value - minimum) / (maximum - minimum)
        knob_x = round(rect.left + fraction * rect.width)
        pygame.draw.line(self.screen, ACCENT, rect.midleft, (knob_x, rect.centery), 4)
        pygame.draw.circle(self.screen, ACCENT, (knob_x, rect.centery), 8)
        pygame.draw.circle(self.screen, CARD, (knob_x, rect.centery), 3)

    def draw_button(
        self, name: str, label: str, rect: pygame.Rect, primary: bool = False
    ) -> None:
        self.buttons[name] = rect
        color = ACCENT if primary else CARD
        label_color = CARD if primary else TEXT
        pygame.draw.rect(self.screen, color, rect, border_radius=10)
        pygame.draw.rect(
            self.screen,
            ACCENT if primary else HAIRLINE,
            rect,
            width=1,
            border_radius=10,
        )
        image = self.heading_font.render(label, True, label_color)
        self.screen.blit(image, image.get_rect(center=rect.center))

    def draw_controls(self, rect: pygame.Rect) -> None:
        self.card(rect, 16)
        top = rect.y + 49
        self.text(self.screen, "Particles", (rect.x + 22, rect.y + 17), SECONDARY_TEXT, self.small_font)
        count_value = str(self.simulation.particle_count)
        count_image = self.value_font.render(count_value, True, TEXT)
        self.screen.blit(count_image, (rect.x + 28, top - 13))
        minus = pygame.Rect(rect.x + 85, top - 15, 30, 30)
        plus = pygame.Rect(rect.x + 122, top - 15, 30, 30)
        self.buttons["minus"] = minus
        self.buttons["plus"] = plus
        for button, glyph in ((minus, "−"), (plus, "+")):
            pygame.draw.rect(self.screen, BACKGROUND, button, border_radius=8)
            pygame.draw.rect(self.screen, HAIRLINE, button, width=1, border_radius=8)
            image = self.heading_font.render(glyph, True, TEXT)
            self.screen.blit(image, image.get_rect(center=button.center))

        slider_start = rect.x + 190
        action_start = rect.right - 226
        gap = 26
        group_width = max(150, (action_start - slider_start - gap) // 2)
        slider_width = group_width - 12
        self.draw_slider(
            "stiffness",
            "Force constant",
            self.simulation.stiffness,
            MIN_K,
            MAX_K,
            pygame.Rect(slider_start, top, slider_width, 1),
            f"{self.simulation.stiffness:.1f}",
        )
        radius_start = slider_start + group_width + gap
        self.draw_slider(
            "radius",
            "Interaction radius",
            self.simulation.interaction_radius,
            MIN_RADIUS,
            MAX_RADIUS,
            pygame.Rect(radius_start, top, slider_width, 1),
            f"{self.simulation.interaction_radius:.0f}",
        )
        reset_rect = pygame.Rect(rect.right - 218, top - 17, 96, 36)
        pause_rect = pygame.Rect(rect.right - 108, top - 17, 90, 36)
        self.draw_button("reset", "Reset", reset_rect)
        self.draw_button(
            "pause",
            "Resume" if self.simulation.paused else "Pause",
            pause_rect,
            primary=True,
        )

    def update_slider(self, name: str, mouse_x: int) -> None:
        rect = self.slider_rects[name]
        fraction = min(1.0, max(0.0, (mouse_x - rect.left) / rect.width))
        if name == "stiffness":
            value = MIN_K + fraction * (MAX_K - MIN_K)
            if value != self.simulation.stiffness:
                self.simulation.stiffness = value
                self.simulation.reset_energy_history()
        else:
            value = MIN_RADIUS + fraction * (MAX_RADIUS - MIN_RADIUS)
            if value != self.simulation.interaction_radius:
                self.simulation.interaction_radius = value
                self.simulation.reset_energy_history()

    def handle_event(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.QUIT:
            return False
        if event.type == pygame.VIDEORESIZE:
            self.screen = pygame.display.set_mode(
                (max(MIN_WIDTH, event.w), max(MIN_HEIGHT, event.h)),
                pygame.RESIZABLE | pygame.SCALED,
            )
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            position = event.pos
            for name, rect in self.buttons.items():
                if rect.collidepoint(position):
                    if name == "minus":
                        self.simulation.set_particle_count(self.simulation.particle_count - 1)
                    elif name == "plus":
                        self.simulation.set_particle_count(self.simulation.particle_count + 1)
                    elif name == "reset":
                        self.simulation.reset()
                    elif name == "pause":
                        self.simulation.paused = not self.simulation.paused
                    return True
            for name, rect in self.slider_rects.items():
                if rect.inflate(0, 20).collidepoint(position):
                    self.dragging_slider = name
                    self.update_slider(name, position[0])
                    return True
            main, _, _ = self.layout()
            if main.collidepoint(position):
                self.dragging_view = True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.dragging_view = False
            self.dragging_slider = None
        elif event.type == pygame.MOUSEMOTION:
            if self.dragging_view:
                self.yaw += event.rel[0] * 0.008
                self.pitch = min(1.25, max(-1.25, self.pitch + event.rel[1] * 0.008))
            if self.dragging_slider:
                self.update_slider(self.dragging_slider, event.pos[0])
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                self.simulation.paused = not self.simulation.paused
            elif event.key == pygame.K_r:
                self.simulation.reset()
            elif event.key == pygame.K_ESCAPE:
                return False
        return True

    def draw(self) -> None:
        self.screen.fill(BACKGROUND)
        self.draw_header()
        main, energy, controls = self.layout()
        self.draw_simulation(main)
        self.draw_energy(energy)
        self.draw_controls(controls)
        pygame.display.flip()

    def run(self) -> None:
        running = True
        accumulator = 0.0
        interval = 1.0 / SIMULATION_HZ
        while running:
            elapsed = min(self.clock.tick(FPS) / 1000.0, 0.1)
            for event in pygame.event.get():
                running = self.handle_event(event)
                if not running:
                    break

            if self.simulation.paused:
                accumulator = 0.0
            else:
                accumulator += elapsed
                while accumulator >= interval:
                    self.simulation.advance()
                    accumulator -= interval

            self.draw()

        pygame.quit()


if __name__ == "__main__":
    PygameApp().run()
