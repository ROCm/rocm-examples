#! /usr/bin/env python3
"""Check Visual Studio files

This script checks the coherency, the uniqueness and the validity of GUIDs and
project names in all Visual Studio files in the project directory.

If a conflict is experienced, the reason is explained with the relevant file
paths.
"""


import os
import re
import sys
import xml.etree.ElementTree as ET
from uuid import UUID
from typing import List, Dict, Tuple, Union
from pathlib import PureWindowsPath
from rich import print


# Shared tools.
class GuidTools:
    error_counter = 0

    def __init__(self) -> None:
        self.directory = os.getcwd()
    # Check GUID format validity.
    def check_guid_validity(self, file: str, guid: str):
        try:
            UUID(guid, version=4)
        except:
            print(f'[red]Incorrect GUID format in [yellow]{file}[/yellow]:[/red]')
            print(f'    {guid}')
            self.error_counter += 1
    def get_guid_validity_errors(self, file: str, guid: str) -> int:
        self.check_guid_validity(file, guid)
        return self.error_counter


# Check all XML files.
class XmlChecker(GuidTools):
    def __init__(self, tag: str, ext: Union[str, None] = None) -> None:
        self.xml_files:List[str] = []
        self.tag_values: Dict[str, List[str]] = {}
        self.error_counter = 0
        self.tag = tag
        super().__init__()
        if ext is not None:
            self.set_xml_files(ext)
            self.set_error_counter()

    # Get XML files.
    def set_xml_files(self, ext: str) -> None:
        for root, _, files in os.walk(self.directory):
            for file in files:
                if file.endswith(ext):
                    self.xml_files.append(os.path.join(root, file))

    # Get tag value from an XML file.
    def set_tag_values(self, file: str) -> None:
        tree = ET.parse(file)
        for elem in tree.iter():
            if elem.text is None:
                continue
            value = elem.text.upper().strip('{}')
            if elem.tag.endswith(self.tag):
                self.error_counter += super().get_guid_validity_errors(file, value)

                if value not in self.tag_values:
                    self.tag_values[value] = [file]
                else:
                    self.tag_values[value].append(file)

    def get_tag_values(self, file: str) -> Dict[str, List[str]]:
        self.set_tag_values(file)
        return self.tag_values

    # Get repeated GUIDs with chosen tag.
    def get_duplicated_tag_values(self) -> Dict[str, List[str]]:
        for file in self.xml_files:
            self.set_tag_values(file)

        # Get repeated GUIDs.
        return {k: v for k, v in self.tag_values.items() if len(v) > 1}

    # Set the number of repeated GUIDs with chosen tag as an error counter.
    def set_error_counter(self) -> None:
        repeated_values = self.get_duplicated_tag_values()
        for value, files in repeated_values.items():
            print(f'[red]The [yellow]{self.tag}[/yellow] value [yellow]{value}[/yellow] is repeated in the following files:[/red]')
            for file in files:
                print(f'    {file}')
            self.error_counter += 1

    # Get error counter.
    def get_error_counter(self) -> int:
        return self.error_counter


# Check all SLN files and the related projects in the root directory.
class SlnChecker(GuidTools):
    error_counter = 0

    def __init__(self) -> None:
        super().__init__()
        self.check_solution_files()

    # Get values from a single project configuration from an SLN file.
    def vars_from_config_match(self, match: Tuple[str, str, str]) -> Union[List[str], Tuple[str, str, str]]:
        guid_raw, config, mode = match
        return guid_raw.upper(), config, mode

    # Get project configurations from SLN file.
    def get_configurations(self, sln_file_content: str) -> List[Tuple[str, str, str]]:
        # Regular expression pattern to match the project configurations.
        global_selection_pattern = r'\s*GlobalSection\(ProjectConfigurationPlatforms\)\s*=\s*postSolution\s*\n(.*?)EndGlobalSection'
        global_selection_matches: List[str] = re.findall(global_selection_pattern, sln_file_content, re.DOTALL)

        # Regular expression pattern to parse a project configuration.
        configuration_pattern = r'\s*{(.*?)}.(.*?)\s*=\s*(.*?)\s*\n'
        return re.findall(configuration_pattern, global_selection_matches[0], re.DOTALL)

    # Validate project configurations from sln file and check whether all
    # configuration has been added.
    def check_configurations(self, sln_file_content: str, reference_guid: str, sln_file_path: str):
        configuration_matches = self.get_configurations(sln_file_content)
        list_of_configurations = [
            ["Debug|x64.ActiveCfg", "Debug|x64"],
            ["Debug|x64.Build.0", "Debug|x64"],
            ["Release|x64.ActiveCfg", "Release|x64"],
            ["Release|x64.Build.0", "Release|x64"]
        ]
        for match in configuration_matches:
            guid, config, mode = self.vars_from_config_match(match)
            if guid == reference_guid:
                try:
                    list_of_configurations.remove([config, mode])
                except ValueError:
                    print(f'[red]Incorrect configuration in [yellow]{sln_file_path}[/yellow]:[/red]')
                    print(f'    {config} = {mode}')
                    self.error_counter += 1
        if len(list_of_configurations):
            print(f'[red]Missing configuration(s) in [yellow]{sln_file_path}[/yellow] for [yellow]{reference_guid}[/yellow]:[/red]')
            for configuration_settings in list_of_configurations:
                print(f'    {configuration_settings[0]} = {configuration_settings[1]}')
            self.error_counter += 1

    # Get project details from SLN file.
    def vars_from_project_match(self, match: Tuple[str, str, str, str]) -> Union[List[str], Tuple[str, str, str]]:
        _, name, path, guid_raw = match
        return guid_raw.upper(), name, path

    # Set path to POSIX format.
    def format_path(self, raw_path: str):
        return PureWindowsPath(raw_path).as_posix()
    
    # Check if the path is a directory.
    def is_dir(self, path: str):
        return not os.path.splitext(path)[1]
    
    # Get matching solution file for project file.
    def get_sln_path(self, project_path: str):
        return os.path.splitext(project_path)[0] + '.sln'
    
    # Get the full path of a solution file in the directory of the project file.
    def get_full_path(self, project_path: str, sln_single_path: str):
        return os.path.dirname(project_path) + '/' + sln_single_path

    # Collect and check project details in an SLN file.
    def parse_sln_file(self, sln_file_path: str) -> Dict[str, Dict[str, str]]:
        with open(sln_file_path, 'r') as file:
            sln_file_content = file.read()
        # Regular expression pattern to match and parse project details.
        pattern = r'Project\("\{(.+?)\}"\)\s*=\s*"(.+?)",\s*"(.+?)",\s*"\{(.+?)\}"'
        matches: List[Tuple[str, str, str, str]] = re.findall(pattern, sln_file_content)

        project_details:Dict[str, Dict[str, str]] = {}
        for match in matches:
            guid, name, path = self.vars_from_project_match(match)
            # Skip subdirectories.
            if self.is_dir(path):
                continue
            project_details[self.format_path(path)]={
                'name': name,
                'guid': guid
            }
            self.check_configurations(sln_file_content, guid, sln_file_path)
        return project_details

    # Get project details for every projects in root SLN file.
    def get_project_details_from_solution(self, project_details:Dict[str, Dict[str, str]]) -> Dict[str, Dict[str, str]]:
        for project_path in project_details:
            # Skip subdirectories.
            if self.is_dir(project_path):
                continue
            sln_path = self.get_sln_path(project_path)
            try:
                single_project_details = self.parse_sln_file(sln_path)
                for sln_single_path in single_project_details:
                    project_detail = single_project_details[sln_single_path]
                    path = self.get_full_path(project_path, sln_single_path)
                    if path not in project_details.keys():
                        print(f'[red]Inconsistent path found in [yellow]{sln_path}[/yellow]:[/red]')
                        print(f'    {sln_single_path}')
                        self.error_counter += 1
                        path = project_path
                    project_details[path]['sln_guid'] = project_detail['guid']
                    project_details[path]['sln_name'] = project_detail['name']
                    project_details[path]['sln_path'] = sln_path

            # Skip if there is no SLN file in embedded projects
            # (see: HIP-Basic\static_host_library).
            except FileNotFoundError:
                pass
        return project_details

    # Check whether the GUIDs are identical in the root and project SLN file
    # and in VCXPROJ file for the same project.
    def check_guid_coherency(self, project_details: Dict[str, Dict[str, str]], sln_file_path: str):
        for project_path in project_details:
            if not os.path.isfile(project_path):
                continue

            project = project_details[project_path]
            project_name = project['name']
            project_guid = project['guid']
            sln_guid = project['sln_guid']
            sln_name = project['sln_name']
            sln_path = project['sln_path']
            xml_checker = XmlChecker(tag='ProjectGuid')
            vcxproj_value = xml_checker.get_tag_values(project_path)
            vcxproj_guid = list(vcxproj_value.keys())[0]

            if project_name != sln_name:
                print(f'[red]Inconsistent project name found in [yellow]{sln_path}[/yellow]:[/red]')
                print(f'    {sln_name}')
                print(f'    The expected project name in [yellow]{sln_file_path}[/yellow]: {project_name}')
                self.error_counter += 1

            if (project_guid == vcxproj_guid and project_guid == sln_guid):
                continue

            print(f'[red]Inconsistent GUID found for [yellow]{project_name}[/yellow] in [yellow]{sln_file_path}[/yellow]:[/red]')
            print(f'    {project_guid}: {sln_file_path}')
            print(f'    {sln_guid}: {sln_path}')
            print(f'    {vcxproj_guid}: {project_path}')
            self.error_counter += 1

    # Check all GUIDs in a root SLN file.
    def check_sln_guids(self, sln_file_path: str):
        project_details = self.parse_sln_file(sln_file_path)
        project_details = self.get_project_details_from_solution(project_details)
        self.check_guid_coherency(project_details, sln_file_path)

    # Check all SLN files in root.
    def check_solution_files(self):
        sln_files = [f for f in os.listdir(self.directory) if f.endswith(".sln")]
        for sln_file in sln_files:
            self.check_sln_guids(sln_file)

    def get_error_counter(self) -> int:
        return self.error_counter


class GuidChecker:
    def __init__(self) -> None:
        # Get repeated GUIDs for UniqueIdentifier in VCXPROJ.FILTERS file.
        self.filters_checker = XmlChecker(tag='UniqueIdentifier', ext='.filters')

        # Get repeated GUIDs for ProjectGuid in VCXPROJ files.
        self.vcxproj_checker = XmlChecker(tag='ProjectGuid', ext='.vcxproj')

        # Get GUID errors in SLN files.
        self.sln_checker = SlnChecker()

        # Set error counter.
        self.set_error_counter()

    # Set error counter as a sum of the errors in different checkers.
    def set_error_counter(self) -> None:
        self.error_counter = self.filters_checker.get_error_counter() \
        + self.vcxproj_checker.get_error_counter() \
        + self.sln_checker.get_error_counter()

    # Get error counter.
    def get_error_counter(self) -> int:
        return self.error_counter


if __name__ == '__main__':
    guid_checker = GuidChecker()
    sys.exit(guid_checker.get_error_counter())
