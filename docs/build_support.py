"""Keep the source-distribution manifest alongside the project documentation."""

from pathlib import Path

from setuptools.command.egg_info import egg_info, manifest_maker


class EggInfo(egg_info):
    """Use docs/MANIFEST.in when setuptools generates the source file list."""

    def find_sources(self):
        template = Path("docs/MANIFEST.in")
        if not template.is_file():
            raise FileNotFoundError("Source distribution template missing: docs/MANIFEST.in")
        maker = manifest_maker(self.distribution)
        maker.ignore_egg_info_dir = self.ignore_egg_info_in_manifest
        maker.template = str(template)
        maker.manifest = str(Path(self.egg_info) / "SOURCES.txt")
        maker.run()
        self.filelist = maker.filelist
