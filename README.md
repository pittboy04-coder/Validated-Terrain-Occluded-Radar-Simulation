# Validated Terrain-Occluded Radar Simulation

Marine radar simulation with terrain height maps, line-of-sight occlusion, and terrain radar returns. Built on top of core modules from the Furuno Radar PPI Simulator.

## Features

- Height-map terrain with bilinear-interpolated elevation queries
- Ray-march occlusion engine for line-of-sight blocking
- Terrain radar returns with shadow casting
- Factory functions for island and ridge terrain generation
- Full radar simulation chain: antenna patterns, detection, weather, clutter, coastlines

## Quick Start

```bash
pip install -r requirements.txt
python main.py
```

## Project Structure

```
radar_sim/
  core/          Range/bearing math, world state, simulation engine
  radar/         Antenna, detection, parameters, system
  environment/   Coastline, clutter, noise, weather, terrain, occlusion
  objects/       Vessel model
main.py          Demo script
```
