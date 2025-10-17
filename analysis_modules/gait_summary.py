"""Gait Summary Model"""
import json
from analysis_modules.utils import convert_numpy_to_native, prepare_output_file


class GaitSummaryAnalysis:
    """Unified container for all analysis outputs from an FBX animation."""
    
    def __init__(self, fbx_path, dopesheet_path, gait_summary=None, chain_conf=None, joint_conf=None, stride_segments=None):
        self.fbx_path = fbx_path
        self.dopesheet_path = dopesheet_path
        self.gait_summary = gait_summary or {}
        self.chain_conf = chain_conf or {}
        self.joint_conf = joint_conf or {}
        self.stride_segments = stride_segments or []
    
    def get_stride_count(self):
        return len(self.stride_segments)
    
    def get_gait_type(self, chain="LeftLeg"):
        return self.gait_summary.get(chain, {}).get("gait_type", "Unknown")
    
    def get_chain_confidence(self, chain):
        return self.chain_conf.get(chain, {}).get("confidence", 0.0)
    
    def to_dict(self):
        data = {
            "fbx_path": self.fbx_path,
            "dopesheet_path": self.dopesheet_path,
            "gait_summary": self.gait_summary,
            "chain_conf": self.chain_conf,
            "joint_conf": {str(k): v for k, v in self.joint_conf.items()},
            "stride_segments": self.stride_segments
        }
        return convert_numpy_to_native(data)
    
    def to_json(self, output_path):
        prepare_output_file(output_path)
        with open(output_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"Gait summary saved to {output_path}")
    
    def __repr__(self):
        return f"<GaitSummaryAnalysis fbx={self.fbx_path} strides={self.get_stride_count()} gait_type={self.get_gait_type()}>"
