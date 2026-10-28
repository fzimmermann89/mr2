"""Replace skipped CUDA placeholders with actual GPU results in a JUnit report."""

import argparse
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

from defusedxml.ElementTree import parse

CUDA_PLACEHOLDER = 'CUDA results are reported separately'


def merge_reports(cpu_path: Path, cuda_path: Path, output_path: Path) -> None:
    """Combine reports, retaining placeholders for CUDA tests that did not run."""
    cpu = parse(cpu_path)
    gpu_cases: dict[tuple[str | None, str | None], list[ET.Element]] = defaultdict(list)
    if cuda_path.is_file():
        for case in parse(cuda_path).iter('testcase'):
            gpu_cases[(case.get('classname'), case.get('name'))].append(case)

    suites = list(cpu.iter('testsuite'))
    for suite in suites:
        for case in list(suite.findall('testcase')):
            skipped = case.find('skipped')
            if skipped is None or skipped.get('message') != CUDA_PLACEHOLDER:
                continue
            key = (case.get('classname'), case.get('name'))
            replacements = gpu_cases.pop(key, [])
            if replacements:
                suite.remove(case)
                suite.extend(replacements)

    if gpu_cases:
        suite = ET.SubElement(cpu.getroot(), 'testsuite', name='CUDA')
        suites.append(suite)
        for cases in gpu_cases.values():
            suite.extend(cases)

    for suite in suites:
        cases = suite.findall('testcase')
        suite.set('tests', str(len(cases)))
        for tag, attribute in [('skipped', 'skipped'), ('failure', 'failures'), ('error', 'errors')]:
            suite.set(attribute, str(sum(case.find(tag) is not None for case in cases)))
        suite.set('time', str(sum(float(case.get('time', '0')) for case in cases)))
    cpu.write(output_path, encoding='utf-8', xml_declaration=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cpu', type=Path)
    parser.add_argument('cuda', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    merge_reports(args.cpu, args.cuda, args.output)
