"""Versioned prompts of the vision-language stage (scripts/object_layers/vlm.py). Change text -> bump VERSION (the on-disk
response cache is keyed by a hash of prompt text + image bytes + model, so edited prompts never reuse stale answers)."""
VERSION = "v2"   # v2: only object cut-outs on black are sent (no context crops of the room)

SYSTEM = ("You are a careful 3D-reconstruction assistant. You look at photographs taken by a small ground robot "
          "(camera about 0.4 m above the floor) and describe the 3D shape of ONE object so that software can rebuild it "
          "from simple solids. Answer with ONE JSON object only: no markdown fences, no text before or after it.")

RECIPE = """The {n_cut} images show the same free-standing object from several robot viewpoints, cut out on a black background
(the cut-out mask can be slightly loose or clip a few pixels).
Measured by LiDAR (may be loose by a few cm, and may wrongly include neighbouring things):
  bounding box  X = {bx:.2f} m (longer horizontal side), Y = {by:.2f} m (shorter horizontal side), Z = {bz:.2f} m (height).

Describe the object as a small set of solid primitives in the box frame: x, y, z in [0, 1] are fractions of the box
(x along the longer horizontal side, y along the shorter one, z up, z = 0 is the floor). Which end of the box is x = 0 is
unknown to you: just be self-consistent; the fit tries the mirror images.
Parts you cannot see still exist: a box seen from the front is a full box; a stack continues to the floor.

Return JSON with exactly these keys:
{{
 "summary": "<one sentence>",
 "rests_on_floor": true|false,
 "detection": {{
   "separate_objects": <int: how many physically separate free-standing objects are inside the cutout; 1 if it is one object or one stack>,
   "partial": true|false,            // the cutout shows only part of the object (cropped by the mask, not by occlusion)
   "extraneous": ["<things in the cutout that are NOT part of the object, e.g. a lamp arm behind it>"],
   "note": "<short>"
 }},
 "primitives": [
   {{"id": "p0",
     "type": "cuboid"|"cylinder"|"wedge"|"plate"|"irregular",
     "center": [x, y],               // footprint centre, box fractions
     "size": [sx, sy, sz],           // extent along x, y, z, box fractions (cylinder: sx = sy = diameter)
     "on": "floor"|"<id of the primitive it stands on>",
     "yaw_deg": 0,                    // rotation about z relative to the box axes
     "axis": "z",                    // cylinder only: "z" upright, "x"/"y" lying
     "low_side": "+x"|"-x"|"+y"|"-y", // wedge only: the side where the sloping face meets its base
     "lean": "+x"|"-x"|"+y"|"-y", "tilt_deg": 0,  // plate only: a thin board standing on the floor, leaning toward this side by tilt_deg from vertical
     "material": "m0"}}
 ],
 "materials": [{{"id": "m0", "description": "<e.g. red brick-pattern cardboard>", "pattern": "plain"|"regular"|"irregular",
                "all_faces_alike": true|false}}],
 "symmetry": {{"mirror_x": true|false, "mirror_y": true|false}},
 "confidence": <0..1 that primitives describe the object well; use "irregular" and low confidence for organic shapes>
}}
Use as few primitives as describe the visible shape (typically 1-4). A-frame = two "plate" primitives leaning toward each other."""

SELECT = """Two candidate 3D reconstructions, A and B, of the object in the real photographs.
The sheet has rows. Rows marked REAL: left = real photo with everything but the object blacked out, middle = candidate A rendered from the same camera, right = candidate B.
Rows marked NOVEL: viewpoints the robot never had (behind / beside / above the object): left = A, right = B.
Judge, in this order: (1) clean, solid, physically plausible unseen sides (flat faces of a box stay flat, no lumps, no
holes, no fringes), (2) outline and proportions matching the real photos, (3) surface appearance on unseen sides
consistent with the seen material, (4) sharpness of the seen sides.
Numeric check against held-out real frames (higher is better): A: silhouette IoU {iou_a}, masked PSNR {psnr_a} dB;
B: silhouette IoU {iou_b}, masked PSNR {psnr_b} dB.
Return JSON: {{"choice": "A"|"B", "reason": "<two sentences>", "defects_A": ["..."], "defects_B": ["..."]}}"""

REVIEW = """Rendered views of a reconstructed scene: objects rebuilt as textured meshes standing on a floor.
Each image is labelled with the object id it centres on and the viewpoint (behind / side / above).
Objects and their sizes (m): {objects}
List concrete, fixable defects only. Allowed edits:
  {{"object": <id>, "edit": "offset", "dx": m, "dy": m, "dz": m}}      // object floats / sinks / is misplaced (image-right = +dx is NOT assumed: use dz for vertical only unless sure)
  {{"object": <id>, "edit": "scale", "sx": f, "sy": f, "sz": f}}        // clearly too wide / tall (0.8..1.25)
  {{"object": <id>, "edit": "material", "issue": "<what>", "fix": "copy_seen"|"mean_colour"}}  // unseen faces whose appearance does not match the object's seen material
Return JSON: {{"edits": [...], "ok_objects": [<ids that look right>], "notes": "<short>"}}. Return an empty edits list if the scene looks right."""
