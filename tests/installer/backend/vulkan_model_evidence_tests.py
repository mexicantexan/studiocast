#!/usr/bin/env python3
"""Full typed packets must carry exact resident-model evidence for each effect."""
import copy
import unittest

from installer_backend_core_tests import (
    active_typed_effect, parse_typed_multi_island_video_status,
    typed_capability_lanes, typed_effect, typed_island,
    typed_multi_island_fixture, typed_pair,
)
from studiocast_installer_backend.typed_video_status import CAPABILITY_EVIDENCE_CODES


class VulkanModelEvidenceTests(unittest.TestCase):
    def packet(self, identifier, name, code_id, token, proofs):
        value = typed_multi_island_fixture()
        value["effects"] = [typed_effect(x["effect"]["id"], x["effect"]["token"])
                            for x in value["effects"]]
        selected = next(x for x in value["effects"] if x["effect"]["id"] == identifier)
        active_typed_effect(selected, 3, "open_vulkan", code_id, token, 1, 11, 5,
                            "vulkan", 5, "open_vulkan_production_kernels")
        semantic = selected["selected_capability"]["semantic"]
        semantic["supports_tracked_vignette_center"] = name == "vignette"
        semantic["evidence"] = [{
            "code": typed_pair(code, CAPABILITY_EVIDENCE_CODES[code]),
            "status": typed_pair(1, "ready"), "reason": typed_pair(0, ""),
            "effect": typed_pair(identifier, name),
            "backend": typed_pair(5, "vulkan"), "satisfied": True,
        } for code in (*range(1, 13), 16, *proofs)]
        value["capabilities"]["lanes"] = typed_capability_lanes(value["effects"])
        value["islands"] = [typed_island(1, "final_visual", 1, 101, 11,
                                       identifier, name, 5, "vulkan", 5,
                                       "open_vulkan_production_kernels")]
        value["authority"]["prepared_route_effect_mask"] = 1 << (identifier - 1)
        value["aggregate"] = {"current_island_count": 1,
                              "transfers": value["islands"][0]["telemetry"]["transfers"]}
        return value

    def test_complete_packets_and_each_missing_required_proof(self):
        cases = (
            (1, "video_noise_removal", 143,
             "open_vulkan_video_noise_removal_production_ready", (50,)),
            (2, "eye_contact", 142,
             "open_vulkan_eye_contact_production_ready", (51, 52, 53, 55)),
            (7, "auto_frame", 146,
             "open_vulkan_auto_frame_production_ready", (51, 54)),
            (8, "vignette", 147,
             "open_vulkan_vignette_tracked_center_production_ready", (19, 51, 54)),
        )
        for identifier, name, code, token, proofs in cases:
            packet = self.packet(identifier, name, code, token, proofs)
            with self.subTest(effect=name, evidence="complete"):
                parsed = parse_typed_multi_island_video_status({"typed_multi_island_video_status": packet})
                self.assertTrue(parsed["valid"], parsed["reason_codes"])
                self.assertFalse(parsed["recommendation_eligible"])
            for missing in (8, 16, *proofs):
                broken = copy.deepcopy(packet)
                effect = next(x for x in broken["effects"] if x["effect"]["id"] == identifier)
                semantic = effect["selected_capability"]["semantic"]
                semantic["evidence"] = [x for x in semantic["evidence"] if x["code"]["id"] != missing]
                broken["capabilities"]["lanes"] = typed_capability_lanes(broken["effects"])
                with self.subTest(effect=name, missing=missing):
                    rejected = parse_typed_multi_island_video_status({"typed_multi_island_video_status": broken})
                    self.assertFalse(rejected["valid"])
                    self.assertEqual(["typed_status.selected_capability_incomplete"], rejected["reason_codes"])

    def test_tracked_flag_cannot_disagree_with_semantic_or_success_code(self):
        cases = ((8, "vignette", 147,
                  "open_vulkan_vignette_tracked_center_production_ready", (19, 51, 54)),
                 (1, "video_noise_removal", 143,
                  "open_vulkan_video_noise_removal_production_ready", (50,)))
        for identifier, name, code, token, proofs in cases:
            packet = self.packet(identifier, name, code, token, proofs)
            effect = next(x for x in packet["effects"] if x["effect"]["id"] == identifier)
            semantic = effect["selected_capability"]["semantic"]
            semantic["supports_tracked_vignette_center"] = not semantic["supports_tracked_vignette_center"]
            packet["capabilities"]["lanes"] = typed_capability_lanes(packet["effects"])
            rejected = parse_typed_multi_island_video_status({"typed_multi_island_video_status": packet})
            self.assertFalse(rejected["valid"], name)


if __name__ == "__main__":
    unittest.main()
