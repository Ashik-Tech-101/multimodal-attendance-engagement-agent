import unittest

import pandas as pd

from attendance_agent.analytics import build_report


class BuildReportTests(unittest.TestCase):
    def test_report_includes_bounded_presence_and_movement_percentages(self) -> None:
        vision = {
            "duration_seconds": 20,
            "video_samples": 10,
            "participants": [
                {
                    "participant_id": "P001",
                    "presence_seconds": 5,
                    "motion_index": 0.25,
                    "last_seen_seconds": 19,
                },
                {
                    "participant_id": "P002",
                    "presence_seconds": 30,
                    "motion_index": 1.5,
                    "last_seen_seconds": 10,
                },
            ],
        }

        report = build_report("Test", vision, pd.DataFrame(), min_presence_seconds=5)

        self.assertEqual(report["participants"][0]["presence_percent"], 25)
        self.assertEqual(report["participants"][0]["activity_percent"], 25)
        self.assertEqual(report["participants"][0]["present"], True)
        self.assertEqual(report["participants"][0]["visibility_status"], "Visible near end")
        self.assertEqual(report["participants"][1]["presence_percent"], 100)
        self.assertEqual(report["participants"][1]["activity_percent"], 100)
        self.assertEqual(report["participants"][1]["visibility_status"], "Not seen at end")


if __name__ == "__main__":
    unittest.main()
