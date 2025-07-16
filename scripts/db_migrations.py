#!/usr/bin/env python3

import argparse
import logging
import os
from pathlib import Path
from typing import Optional

from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from alembic.util import CommandError

from config.settings import Settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseMigrator:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.alembic_cfg = self._create_alembic_config()

    def _create_alembic_config(self) -> Config:
        """Create and configure Alembic Config object."""
        # Get the directory containing this script
        migrations_dir = Path(__file__).parent.parent / 'infrastructure' / 'migrations'
        if not migrations_dir.exists():
            migrations_dir.mkdir(parents=True)

        # Create alembic.ini if it doesn't exist
        alembic_ini = migrations_dir / 'alembic.ini'
        if not alembic_ini.exists():
            self._initialize_alembic(migrations_dir)

        # Create and configure alembic Config
        config = Config(str(alembic_ini))
        config.set_main_option('script_location', str(migrations_dir))
        config.set_main_option('sqlalchemy.url', self.settings.database_url)
        return config

    def _initialize_alembic(self, migrations_dir: Path) -> None:
        """Initialize a new Alembic environment."""
        template_dir = Path(__file__).parent / 'templates' / 'alembic'
        if not template_dir.exists():
            raise RuntimeError("Alembic templates directory not found")

        # Create basic alembic.ini
        with open(migrations_dir / 'alembic.ini', 'w') as f:
            f.write("""[alembic]
# path to migration scripts
script_location = %(here)s

# template used to generate migration files
file_template = %%(rev)s_%%(slug)s

# timezone to use when rendering the date
timezone = UTC

[loggers]
keys = root,sqlalchemy,alembic

[handlers]
keys = console

[formatters]
keys = generic

[logger_root]
level = WARN
handlers = console
qualname =

[logger_sqlalchemy]
level = WARN
handlers =
qualname = sqlalchemy.engine

[logger_alembic]
level = INFO
handlers =
qualname = alembic

[handler_console]
class = StreamHandler
args = (sys.stderr,)
level = NOTSET
formatter = generic

[formatter_generic]
format = %(levelname)-5.5s [%(name)s] %(message)s
datefmt = %H:%M:%S""")

    def create_migration(self, message: str) -> None:
        """Create a new migration revision."""
        try:
            command.revision(self.alembic_cfg, message=message, autogenerate=True)
            logger.info(f"Created new migration with message: {message}")
        except CommandError as e:
            logger.error(f"Failed to create migration: {str(e)}")
            raise

    def upgrade(self, revision: str = 'head') -> None:
        """Upgrade database to specified revision."""
        try:
            command.upgrade(self.alembic_cfg, revision)
            logger.info(f"Successfully upgraded database to: {revision}")
        except CommandError as e:
            logger.error(f"Failed to upgrade database: {str(e)}")
            raise

    def downgrade(self, revision: str) -> None:
        """Downgrade database to specified revision."""
        try:
            command.downgrade(self.alembic_cfg, revision)
            logger.info(f"Successfully downgraded database to: {revision}")
        except CommandError as e:
            logger.error(f"Failed to downgrade database: {str(e)}")
            raise

    def current(self) -> None:
        """Show current revision."""
        command.current(self.alembic_cfg)

    def history(self) -> None:
        """Show migration history."""
        command.history(self.alembic_cfg)

def main():
    parser = argparse.ArgumentParser(description="Database migration tool")
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Create migration
    create_parser = subparsers.add_parser('create', help='Create a new migration')
    create_parser.add_argument('message', help='Migration message')

    # Upgrade
    upgrade_parser = subparsers.add_parser('upgrade', help='Upgrade database')
    upgrade_parser.add_argument('--revision', default='head',
                               help='Target revision (default: head)')

    # Downgrade
    downgrade_parser = subparsers.add_parser('downgrade', help='Downgrade database')
    downgrade_parser.add_argument('revision', help='Target revision')

    # Current and History
    subparsers.add_parser('current', help='Show current revision')
    subparsers.add_parser('history', help='Show migration history')

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return 1

    try:
        settings = Settings()
        migrator = DatabaseMigrator(settings)

        if args.command == 'create':
            migrator.create_migration(args.message)
        elif args.command == 'upgrade':
            migrator.upgrade(args.revision)
        elif args.command == 'downgrade':
            migrator.downgrade(args.revision)
        elif args.command == 'current':
            migrator.current()
        elif args.command == 'history':
            migrator.history()

        return 0

    except Exception as e:
        logger.error(f"Error during migration: {str(e)}")
        return 1

if __name__ == '__main__':
    exit(main())