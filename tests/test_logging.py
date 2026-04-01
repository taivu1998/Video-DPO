import os
import tempfile
import unittest

from src.logging_utils import setup_logger


def _cleanup_logger(logger):
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()


class LoggingTests(unittest.TestCase):
    def test_setup_logger_is_idempotent_without_log_dir(self):
        logger = setup_logger("video_dpo_test_logger")
        logger = setup_logger("video_dpo_test_logger")
        self.addCleanup(_cleanup_logger, logger)
        self.assertEqual(len(logger.handlers), 1)

    def test_setup_logger_replaces_handlers_for_same_log_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = setup_logger("video_dpo_file_logger", tmpdir)
            logger = setup_logger("video_dpo_file_logger", tmpdir)
            self.addCleanup(_cleanup_logger, logger)

            self.assertEqual(len(logger.handlers), 2)
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "train.log")))


if __name__ == "__main__":
    unittest.main()
