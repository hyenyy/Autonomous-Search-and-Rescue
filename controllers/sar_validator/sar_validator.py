"""Independent Webots ground-truth observer; never feeds data to navigation."""
import json
import math
import os
from pathlib import Path
import sys
from collections import deque

from controller import Supervisor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from sar.runtime import write_json


def main():
    robot = Supervisor()
    timestep = int(robot.getBasicTimeStep())
    children = robot.getRoot().getField('children')
    nodes = [children.getMFNode(i) for i in range(children.getCount())]
    target = next(n for n in nodes if n.getTypeName() == 'TurtleBot3Burger')
    apples = [n for n in nodes if n.getTypeName() == 'RedApple']
    start = target.getPosition()[:2]
    out = Path(os.environ.get('SAR_OUTPUT_DIR', ROOT / 'out'))
    mins = [float('inf')] * len(apples)
    last = -1
    min_upright = 1.0
    done_since = None
    truth = deque(maxlen=500)
    while robot.step(timestep) != -1:
        now = robot.getTime()
        p = target.getPosition()
        truth.append((now, p[:]))
        orientation = target.getOrientation()
        min_upright = min(min_upright, orientation[8])
        for i, apple in enumerate(apples):
            a = apple.getPosition()
            mins[i] = min(mins[i], math.hypot(p[0] - a[0], p[1] - a[1]))
        if now - last < 1:
            continue
        last = now
        try:
            status = json.loads((out / 'live_status.json').read_text(encoding='utf-8'))
        except (OSError, ValueError):
            status = {}
        home = math.hypot(p[0] - start[0], p[1] - start[1])
        pose = status.get('pose')
        sample = min(truth, key=lambda row: abs(row[0] - status.get('sim_time', now)))
        synced = pose is not None and abs(sample[0] - status.get('sim_time', now)) < .1
        report = dict(sim_time=now, run_id=status.get('run_id'), true_position=p, home_distance=home,
                      min_target_distances=mins, min_upright=min_upright,
                      current_upright=orientation[8], controller_state=status.get('state'),
                      controller_success=status.get('success', False),
                      visited=status.get('visited', 0),
                      pose_error=math.hypot(sample[1][0]-pose[0], sample[1][1]-pose[1]) if synced else None)
        write_json(out / 'ground_truth.json', report)
        with (out / 'ground_truth.jsonl').open('a', encoding='utf-8') as log:
            log.write(json.dumps(report) + '\n')
        if status.get('state') == 'DONE':
            if done_since is None:
                done_since = now
        if (done_since is not None and now - done_since >= 2) or now >= 1500 or status.get('state') == 'ERROR':
            report['passed'] = bool(status.get('success') and len(mins) == 2
                                    and max(mins) < .75 and home < .4)
            write_json(out / 'validation_result.json', report)
            print('[validator] ' + json.dumps(report), flush=True)
            robot.simulationQuit(0 if report['passed'] else 1)
            return


if __name__ == '__main__':
    main()
