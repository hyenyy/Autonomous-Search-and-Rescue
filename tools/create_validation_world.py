"""Create an ignored validation copy of the real apartment with an observer."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / 'worlds/apartment_sar.wbt'
dest = ROOT / 'worlds/apartment_sar_validation.wbt'
world = source.read_text(encoding='utf-8')
world += '\nRobot {\n  supervisor TRUE\n  name "SAR independent validator"\n  controller "sar_validator"\n}\n'
dest.write_text(world, encoding='utf-8')
(ROOT / 'controllers/sar_validator/runtime.ini').write_text(
    '[python]\nCOMMAND = ' + (ROOT / '.venv/Scripts/python.exe').as_posix() + '\n', encoding='utf-8')
print(dest)
