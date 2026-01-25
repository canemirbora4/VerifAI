"""
Provenance Analysis
====================

Analyzes digital provenance and content credentials:
- C2PA (Coalition for Content Provenance and Authenticity) manifests
- Content Credentials verification
- Digital signature validation
- Edit history tracking

C2PA is an open standard that embeds cryptographically signed metadata
into media files, providing a verifiable chain of custody from creation
through edits.

Key Concepts:
- **Manifest**: Container for provenance data embedded in the file
- **Claim**: Signed assertion about the content (who created it, how)
- **Ingredient**: Reference to source material used in creation
- **Action**: Edit operation performed on the content

Real photos from C2PA-enabled cameras have valid manifests;
AI-generated images typically have no provenance or invalid claims.

References:
- C2PA Specification: https://c2pa.org/specifications/
- Content Authenticity Initiative: https://contentauthenticity.org/
"""

from dataclasses import dataclass, field
from typing import Optional, Union, Any
from pathlib import Path
from datetime import datetime
import json
import struct

from PIL import Image
from loguru import logger


@dataclass
class C2PAClaim:
    """
    A C2PA claim (signed assertion about content).
    
    Attributes:
        claim_generator: Software/device that generated the claim
        title: Claim title
        format: Media format
        instance_id: Unique identifier
        signature_info: Signature details
        actions: List of actions/edits
        ingredients: Source materials used
    """
    claim_generator: str = ""
    title: str = ""
    format: str = ""
    instance_id: str = ""
    signature_info: dict = field(default_factory=dict)
    actions: list[dict] = field(default_factory=list)
    ingredients: list[dict] = field(default_factory=list)
    is_valid: bool = False
    validation_errors: list[str] = field(default_factory=list)


@dataclass
class ProvenanceFeatures:
    """
    Container for provenance analysis results.
    
    Attributes:
        has_c2pa: Whether C2PA manifest was found
        has_valid_signature: Whether signatures are valid
        claims: List of C2PA claims found
        creation_tool: Tool that created the content
        creation_date: When content was created
        edit_history: List of edits performed
        provenance_score: Score indicating authenticity (0-1)
        trust_indicators: Positive trust signals found
        risk_indicators: Risk signals found
    """
    
    # C2PA specific
    has_c2pa: bool = False
    has_valid_signature: bool = False
    claims: list[C2PAClaim] = field(default_factory=list)
    
    # General provenance
    creation_tool: Optional[str] = None
    creation_date: Optional[datetime] = None
    edit_history: list[dict] = field(default_factory=list)
    
    # Scoring
    provenance_score: float = 0.5  # 0 = no provenance, 1 = verified authentic
    
    # Indicators
    trust_indicators: list[str] = field(default_factory=list)
    risk_indicators: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "has_c2pa": self.has_c2pa,
            "has_valid_signature": self.has_valid_signature,
            "num_claims": len(self.claims),
            "creation_tool": self.creation_tool,
            "creation_date": self.creation_date.isoformat() if self.creation_date else None,
            "edit_count": len(self.edit_history),
            "provenance_score": round(self.provenance_score, 4),
            "trust_indicators": self.trust_indicators,
            "risk_indicators": self.risk_indicators,
        }
    
    @property
    def is_verified(self) -> bool:
        """Check if content has verified provenance."""
        return self.has_c2pa and self.has_valid_signature


class ProvenanceAnalyzer:
    """
    Analyzes digital provenance and content credentials.
    
    Usage:
        analyzer = ProvenanceAnalyzer()
        
        # Analyze single image
        features = analyzer.analyze("photo.jpg")
        
        if features.is_verified:
            print(f"Verified by: {features.creation_tool}")
        else:
            print(f"Risk indicators: {features.risk_indicators}")
    """
    
    # Known AI generation tools (for detection)
    AI_GENERATORS = {
        "midjourney", "stable diffusion", "dall-e", "dalle",
        "openai", "stability ai", "runway", "leonardo",
        "firefly", "imagen", "parti", "muse", "sdxl",
        "automatic1111", "comfyui", "invoke ai", "diffusers",
    }
    
    # Known legitimate camera/editing software
    TRUSTED_TOOLS = {
        # Cameras
        "canon", "nikon", "sony", "fujifilm", "panasonic",
        "olympus", "leica", "hasselblad", "phase one",
        # Mobile
        "apple", "iphone", "google pixel", "samsung galaxy",
        # Editors
        "adobe photoshop", "adobe lightroom", "capture one",
        "affinity photo", "gimp", "darktable",
    }
    
    def __init__(self):
        """Initialize the provenance analyzer."""
        pass
    
    def analyze(
        self,
        source: Union[str, Path, bytes],
    ) -> ProvenanceFeatures:
        """
        Analyze provenance of a media file.
        
        Args:
            source: File path or bytes
            
        Returns:
            ProvenanceFeatures with analysis results
        """
        features = ProvenanceFeatures()
        
        # Load file
        if isinstance(source, bytes):
            data = source
            path = None
        else:
            path = Path(source)
            if not path.exists():
                logger.warning(f"File not found: {path}")
                return features
            
            with open(path, "rb") as f:
                data = f.read()
        
        # Check for C2PA manifest
        c2pa_result = self._check_c2pa(data)
        features.has_c2pa = c2pa_result["found"]
        features.claims = c2pa_result.get("claims", [])
        features.has_valid_signature = c2pa_result.get("valid_signature", False)
        
        # Check for XMP metadata (Adobe, etc.)
        xmp_result = self._check_xmp(data)
        if xmp_result.get("creation_tool"):
            features.creation_tool = xmp_result["creation_tool"]
        if xmp_result.get("creation_date"):
            features.creation_date = xmp_result["creation_date"]
        features.edit_history = xmp_result.get("history", [])
        
        # Analyze indicators
        features.trust_indicators = self._find_trust_indicators(features, data)
        features.risk_indicators = self._find_risk_indicators(features, data)
        
        # Compute provenance score
        features.provenance_score = self._compute_score(features)
        
        return features
    
    def _check_c2pa(self, data: bytes) -> dict:
        """
        Check for C2PA manifest in file.
        
        C2PA data is stored in:
        - JPEG: APP11 marker segment (0xFFEB) with "C2PA" identifier
        - PNG: c2pa chunk
        - TIFF/HEIF: Box structure
        """
        result = {
            "found": False,
            "claims": [],
            "valid_signature": False,
        }
        
        # Check for C2PA magic bytes
        c2pa_markers = [
            b"c2pa",
            b"C2PA",
            b"jumbf",  # JUMBF box (used by C2PA)
            b"c2pa_manifest",
        ]
        
        for marker in c2pa_markers:
            if marker in data:
                result["found"] = True
                break
        
        if not result["found"]:
            # Check JPEG APP11 marker
            if data[:2] == b'\xff\xd8':  # JPEG magic
                result["found"] = self._check_jpeg_c2pa(data)
            # Check PNG c2pa chunk
            elif data[:8] == b'\x89PNG\r\n\x1a\n':  # PNG magic
                result["found"] = self._check_png_c2pa(data)
        
        if result["found"]:
            # Try to parse claims
            claims = self._parse_c2pa_claims(data)
            result["claims"] = claims
            
            # Validate signature (simplified - real validation requires crypto)
            result["valid_signature"] = self._validate_c2pa_signature(data, claims)
        
        return result
    
    def _check_jpeg_c2pa(self, data: bytes) -> bool:
        """Check JPEG for C2PA APP11 segment."""
        pos = 2  # Skip SOI
        
        while pos < len(data) - 4:
            if data[pos] != 0xFF:
                break
            
            marker = data[pos + 1]
            
            # APP11 = 0xEB
            if marker == 0xEB:
                length = struct.unpack(">H", data[pos+2:pos+4])[0]
                segment = data[pos+4:pos+2+length]
                
                if b"c2pa" in segment.lower() or b"jumbf" in segment:
                    return True
            
            # Move to next marker
            if marker in (0xD8, 0xD9, 0x01) or 0xD0 <= marker <= 0xD7:
                pos += 2
            else:
                length = struct.unpack(">H", data[pos+2:pos+4])[0]
                pos += 2 + length
        
        return False
    
    def _check_png_c2pa(self, data: bytes) -> bool:
        """Check PNG for c2pa chunk."""
        pos = 8  # Skip signature
        
        while pos < len(data) - 12:
            length = struct.unpack(">I", data[pos:pos+4])[0]
            chunk_type = data[pos+4:pos+8]
            
            if chunk_type in (b"c2pa", b"caBX", b"jumb"):
                return True
            
            # Move to next chunk
            pos += 12 + length  # length + type + data + CRC
        
        return False
    
    def _parse_c2pa_claims(self, data: bytes) -> list[C2PAClaim]:
        """
        Parse C2PA claims from manifest.
        
        Note: Full parsing requires JUMBF/CBOR parsing.
        This is a simplified version that extracts key fields.
        """
        claims = []
        
        # Look for claim-related patterns
        patterns = [
            (b'"claim_generator"', "claim_generator"),
            (b'"dc:title"', "title"),
            (b'"dc:format"', "format"),
            (b'"xmpMM:InstanceID"', "instance_id"),
        ]
        
        # Simple extraction (would need proper CBOR parsing for production)
        claim = C2PAClaim()
        
        for pattern, field_name in patterns:
            pos = data.find(pattern)
            if pos != -1:
                # Extract value (rough extraction)
                start = data.find(b'"', pos + len(pattern)) + 1
                end = data.find(b'"', start)
                if start > 0 and end > start:
                    value = data[start:end].decode("utf-8", errors="ignore")
                    setattr(claim, field_name, value)
        
        # Check for action patterns
        if b'"c2pa.created"' in data:
            claim.actions.append({"action": "c2pa.created"})
        if b'"c2pa.edited"' in data:
            claim.actions.append({"action": "c2pa.edited"})
        if b'"c2pa.filtered"' in data:
            claim.actions.append({"action": "c2pa.filtered"})
        
        # Check for AI-related actions
        if b'"c2pa.ai_generated"' in data or b'"c2pa.composed"' in data:
            claim.actions.append({"action": "c2pa.ai_generated"})
        
        if claim.claim_generator or claim.actions:
            claims.append(claim)
        
        return claims
    
    def _validate_c2pa_signature(
        self,
        data: bytes,
        claims: list[C2PAClaim],
    ) -> bool:
        """
        Validate C2PA signature.
        
        Note: Full validation requires certificate chain verification.
        This is a simplified check.
        """
        # Check for signature-related data
        sig_markers = [
            b"-----BEGIN CERTIFICATE-----",
            b"sigTst",  # Signature timestamp
            b"cose-signature",
        ]
        
        has_signature = any(marker in data for marker in sig_markers)
        
        # For now, assume valid if signature data present
        # (Real implementation would verify the crypto)
        return has_signature
    
    def _check_xmp(self, data: bytes) -> dict:
        """
        Check for XMP metadata.
        
        XMP (Extensible Metadata Platform) is Adobe's metadata standard,
        often containing creation tool and edit history.
        """
        result = {
            "creation_tool": None,
            "creation_date": None,
            "history": [],
        }
        
        # Find XMP packet
        xmp_start = data.find(b"<?xpacket begin")
        if xmp_start == -1:
            xmp_start = data.find(b"<x:xmpmeta")
        
        if xmp_start == -1:
            return result
        
        xmp_end = data.find(b"<?xpacket end", xmp_start)
        if xmp_end == -1:
            xmp_end = data.find(b"</x:xmpmeta>", xmp_start)
        
        if xmp_end == -1:
            return result
        
        xmp_data = data[xmp_start:xmp_end + 100]
        xmp_str = xmp_data.decode("utf-8", errors="ignore")
        
        # Extract creation tool
        tool_patterns = [
            ("xmp:CreatorTool", "creation_tool"),
            ("tiff:Software", "creation_tool"),
            ("photoshop:History", "history_marker"),
        ]
        
        for pattern, key in tool_patterns:
            if pattern in xmp_str:
                # Simple extraction
                start = xmp_str.find(pattern)
                value_start = xmp_str.find(">", start) + 1
                value_end = xmp_str.find("<", value_start)
                
                if value_start > 0 and value_end > value_start:
                    value = xmp_str[value_start:value_end].strip()
                    if key == "creation_tool":
                        result["creation_tool"] = value
        
        # Extract date
        date_patterns = ["xmp:CreateDate", "xmp:ModifyDate", "exif:DateTimeOriginal"]
        for pattern in date_patterns:
            if pattern in xmp_str:
                start = xmp_str.find(pattern)
                value_start = xmp_str.find(">", start) + 1
                value_end = xmp_str.find("<", value_start)
                
                if value_start > 0 and value_end > value_start:
                    date_str = xmp_str[value_start:value_end].strip()
                    try:
                        # Parse ISO date
                        result["creation_date"] = datetime.fromisoformat(
                            date_str.replace("Z", "+00:00")
                        )
                    except:
                        pass
                    break
        
        # Check for edit history
        if "stEvt:action" in xmp_str:
            # Count edit actions
            edit_count = xmp_str.count("stEvt:action")
            for i in range(edit_count):
                result["history"].append({"action": f"edit_{i+1}"})
        
        return result
    
    def _find_trust_indicators(
        self,
        features: ProvenanceFeatures,
        data: bytes,
    ) -> list[str]:
        """Find positive trust indicators."""
        indicators = []
        
        # C2PA indicators
        if features.has_c2pa:
            indicators.append("C2PA manifest present")
            if features.has_valid_signature:
                indicators.append("Valid C2PA signature")
        
        # Tool indicators
        if features.creation_tool:
            tool_lower = features.creation_tool.lower()
            for trusted in self.TRUSTED_TOOLS:
                if trusted in tool_lower:
                    indicators.append(f"Created with trusted tool: {features.creation_tool}")
                    break
        
        # Date indicators
        if features.creation_date:
            indicators.append("Creation date present")
        
        # Check for camera-specific markers in raw data
        camera_markers = [
            (b"Canon", "Canon camera metadata"),
            (b"NIKON", "Nikon camera metadata"),
            (b"SONY", "Sony camera metadata"),
            (b"Apple", "Apple device metadata"),
        ]
        
        for marker, description in camera_markers:
            if marker in data:
                indicators.append(description)
                break
        
        return indicators
    
    def _find_risk_indicators(
        self,
        features: ProvenanceFeatures,
        data: bytes,
    ) -> list[str]:
        """Find risk indicators suggesting AI generation or manipulation."""
        indicators = []
        
        # No provenance at all
        if not features.has_c2pa and not features.creation_tool:
            indicators.append("No provenance metadata")
        
        # C2PA present but invalid
        if features.has_c2pa and not features.has_valid_signature:
            indicators.append("C2PA signature invalid or missing")
        
        # Check for AI generation markers
        if features.creation_tool:
            tool_lower = features.creation_tool.lower()
            for ai_tool in self.AI_GENERATORS:
                if ai_tool in tool_lower:
                    indicators.append(f"AI generation tool detected: {features.creation_tool}")
                    break
        
        # Check for AI markers in claims
        for claim in features.claims:
            for action in claim.actions:
                if "ai" in str(action).lower():
                    indicators.append("C2PA AI generation action declared")
                    break
        
        # Check raw data for AI markers
        ai_markers = [
            (b"stable-diffusion", "Stable Diffusion marker"),
            (b"midjourney", "Midjourney marker"),
            (b"DALL-E", "DALL-E marker"),
            (b"ComfyUI", "ComfyUI marker"),
            (b"automatic1111", "A1111 marker"),
            (b"Dream by WOMBO", "WOMBO marker"),
        ]
        
        for marker, description in ai_markers:
            if marker.lower() in data.lower():
                indicators.append(description)
        
        # Check for suspiciously clean metadata (AI often has minimal/no EXIF)
        exif_markers = [b"Exif", b"EXIF", b"GPS"]
        has_exif = any(marker in data for marker in exif_markers)
        
        if not has_exif and len(data) > 50000:  # Large file with no EXIF
            indicators.append("No EXIF data (unusual for photos)")
        
        return indicators
    
    def _compute_score(self, features: ProvenanceFeatures) -> float:
        """
        Compute overall provenance score.
        
        0.0 = Definitely no authentic provenance
        0.5 = Unknown/uncertain
        1.0 = Verified authentic provenance
        """
        score = 0.5  # Start neutral
        
        # Positive adjustments
        if features.has_c2pa:
            score += 0.15
            if features.has_valid_signature:
                score += 0.2
        
        if features.creation_tool:
            tool_lower = features.creation_tool.lower()
            for trusted in self.TRUSTED_TOOLS:
                if trusted in tool_lower:
                    score += 0.1
                    break
        
        if features.creation_date:
            score += 0.05
        
        # Negative adjustments
        for indicator in features.risk_indicators:
            if "AI" in indicator or "ai" in indicator.lower():
                score -= 0.3
            elif "No provenance" in indicator:
                score -= 0.15
            elif "invalid" in indicator.lower():
                score -= 0.2
            else:
                score -= 0.1
        
        return max(0.0, min(1.0, score))


# Convenience function
def analyze_provenance(
    source: Union[str, Path, bytes],
) -> ProvenanceFeatures:
    """
    Analyze provenance of a media file.
    
    Args:
        source: File path or bytes
        
    Returns:
        ProvenanceFeatures object
    """
    analyzer = ProvenanceAnalyzer()
    return analyzer.analyze(source)
