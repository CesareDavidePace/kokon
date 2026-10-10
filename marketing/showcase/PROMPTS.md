# Showcase images

Every image here was made by Kokon itself (Qwen-Image 2.1 Turbo Q4_K, 8 steps, CFG 1) on an
RTX 5070 Ti through the Kokon MCP server on 2026-10-10. No stock photos, so no third-party
rights. Originals (PNG, full size) are in `%USERPROFILE%\Pictures\Kokon\marketing`; these JPEGs
are resized to at most 2400 px.

| Pair | Step | Tool | Prompt | Seed | Quality | Time |
| --- | --- | --- | --- | --- | --- | --- |
| Remove | `street` | generate 16:9 | Editorial photograph of a sunny cobblestone street in a small Italian coastal town, a pastel yellow café with outdoor wooden tables and potted lemon trees, a red vintage scooter parked in front of the café, warm late-morning light, 35mm, shallow depth of field, highly detailed, realistic | 1101 | quality | 46 s |
| | `street-removed` | inpaint, remove, `street-mask.png` | the red scooter | 7 | quality | 33 s |
| Add | `street-cat` | inpaint, add, `street-add-mask.png` on `street-removed` | a ginger cat sitting on the cobblestones, looking at the camera | 13 | quality | 31 s |
| Relight | `cabin` | generate 16:9 | Photograph of a small wooden cabin with a stone chimney on the shore of a calm alpine lake, pine forest and tall mountains behind, clear summer midday, crisp reflections on the water, wide angle, highly detailed, realistic landscape photography | 2202 | quality | 39 s |
| | `cabin-winter` | edit | Make it a snowy winter night: snow on the roof, trees and shore, the lake partly frozen, warm golden light glowing from the cabin windows, stars in the sky. Keep the composition unchanged. | 3 | quality | 53 s |
| Text | `bakery` | generate 4:3 | Photograph of a charming bakery storefront in Paris, dark green painted wooden facade, large gold lettering sign above the window that reads "BOULANGERIE", window display full of croissants and baguettes, flower boxes, soft morning light, realistic, highly detailed | 3303 | quality | 39 s |
| | `bakery-text` | edit | Change the gold sign text to "KOKON CAFÉ" in the same gold lettering style. Keep everything else unchanged. | 5 | quality | 54 s |
| Restyle | `potter` | generate 3:4 | Portrait photograph of a young ceramic artist in her sunlit studio, holding a freshly glazed blue bowl, shelves of handmade pottery behind her, linen apron, natural window light, warm and friendly expression, 85mm lens, realistic, highly detailed | 4404 | quality | 38 s |
| | `potter-watercolor` | edit | Turn it into a hand-painted watercolor illustration with soft ink outlines and visible paper texture. Keep the person, pose and composition the same. | 11 | quality | 55 s |
| Replace | `room` | generate 4:3 | Interior design photograph of a bright Scandinavian living room, a mustard yellow armchair next to a large window, light oak floor, white walls, a floor lamp, plants, a woven rug, soft daylight, realistic, highly detailed | 5505 | quality | 38 s |
| | `room-sofa` | edit | Replace the yellow armchair with a deep green velvet three-seat sofa with two cushions, and hang a large framed abstract painting on the wall above it. Keep the room, light and everything else unchanged. | 21 | quality | 54 s |
| Expand | `lighthouse` | generate 1:1 | Photograph of a white and red lighthouse on a rocky cliff above a turquoise sea, golden sunset light, gentle waves, a few seagulls, realistic, highly detailed, coastal landscape photography | 6606 | balanced | 22 s |
| | `lighthouse-wide` | outpaint, 40 % left and right | continue the rocky coastline and the turquoise sea at golden sunset | 9 | balanced | 55 s |
| Product | `sneaker` | generate 1:1 | Product photograph of a minimalist white leather sneaker on a plain light grey background, studio lighting, soft shadow, e-commerce catalog shot, sharp details, realistic | 7707 | balanced | 22 s |
| | `sneaker-ad` | edit | Place the sneaker on a wet black stone surface in a moody studio with a soft orange rim light and gentle reflections, product advertising shot. Keep the sneaker exactly the same. | 17 | balanced | 25 s |
| Hero | `hero` | generate 16:9 | Cinematic macro photograph of a luminous blue morpho butterfly resting on a silk cocoon hanging from a mossy branch, morning dew drops, soft warm backlight, dark green bokeh background, ultra detailed, realistic | 8808 | quality | 39 s |

Every result is the first try with that seed; nothing was cherry-picked.
