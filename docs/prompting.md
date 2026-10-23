# How to write prompts in Kokon

> **The visual version is at [kokon.app/prompt-guide](https://kokon.app/prompt-guide/)**: every rule
> with the vague prompt and the clear one side by side. This page is the text reference.

Kokon runs Qwen-Image 2.1 Turbo. It follows what you write closely, so a clear description gets
a better picture than a clever one. This page shows what works, with examples tested in Kokon
itself (RTX 5070 Ti, Fast quality, fixed seeds, first try).

## The short version

1. **Describe the scene, not just the subject.** Say who or what, what they are doing, where,
   and the light. "A girl riding a bicycle" works; "a young woman riding a vintage blue bicycle
   along a seaside promenade, late afternoon golden light, full body" gets you *that* picture.
2. **Say what you want, never what you don't want.** "No laptop" puts a laptop on the desk.
   Describe the desk as "clean and empty except for a plant and a cup" instead.
3. **Put text in quotes.** `a neon sign that reads "Luna Caffè"`. Keep it short; for small text
   use the Quality setting.
4. **For edits, name the thing and the change.** "Replace the yellow armchair with a green
   velvet sofa" beats "make it nicer". Vague prompts ("make it better") return the same image.
5. **Any language works.** Italian, English and others give the same result when the prompt is
   descriptive. Short prompts lose detail in every language.

If a result is close but not right, press **Regenerate** (new seed) before rewriting the prompt.

---

## Creating a new image

### A simple formula

> **[type of image] of [subject] [doing what], [where], [light / time], [framing], [style]**

You don't need every part. Add the ones you care about.

| Instead of | Write |
| --- | --- |
| a monkey playing basketball | Photograph of a monkey dribbling a basketball on an outdoor court, full body, sunny afternoon |
| a cat | Photograph of a ginger cat asleep on a sunlit windowsill, potted plants around, soft morning light, close-up |
| a girl riding a bicycle | Photograph of a young woman riding a vintage blue bicycle along a seaside promenade, her hair blowing in the wind, a basket of flowers on the handlebar, golden late-afternoon light, shot from the side, full body |
| a coffee shop | Photograph of a small coffee shop at night, a neon sign above the door that reads "Luna Caffè", warm light from the window, rainy street |

Why it matters: with a short prompt the model draws the subject and fills the rest with the
most ordinary choice. "A monkey playing basketball" gave a monkey sitting on a post, no ball.
Adding "on a court, ball in hand, full body" gave the full scene.

### Long prompts are fine

The model handles long descriptions well. This one got every element right, cat included:

> Cozy photograph of a small Japanese ramen shop interior at night seen from the counter. On the
> left an old chef with a white headband pours broth into a bowl, steam rising. On the right two
> customers in winter jackets sit on wooden stools, one laughing. Paper lanterns hang from the
> ceiling, handwritten menu boards on the wall, a cat sleeps on a shelf in the background. Warm
> tungsten light, slightly grainy film look, 35mm, shallow depth of field.

Write in plain sentences. Put the most important things first.

### Positions and counts

Left, right, middle, behind, in the background, in the corner: they work.

> Still life photograph on a wooden kitchen table: three red apples on the left, a single green
> pear on the right, a white ceramic jug in the middle behind them, soft window light from the left

Result: exactly three apples, pear on the right, jug in the middle.

### Camera and framing

| You want | Words that work |
| --- | --- |
| The whole person | full body, wide shot |
| A face | close-up portrait, 85mm |
| Seen from above | bird's-eye view, straight down, flat lay |
| Seen from below | low-angle photograph looking up at… |
| Background blur | shallow depth of field |
| Shape of the image | the aspect menu, or put it in the prompt ("16:9 photograph of…") |

### Text in the image

Qwen-Image is good at text. Put the exact words in quotes and say where they go.

> Minimal concert poster, dark blue background, a large white crescent moon in the center. Only
> two lines of text: the title "NIGHT JAZZ" in big bold letters at the top, and "24 OCTOBER" in
> small letters at the bottom. Flat graphic style.

- Big titles come out right almost every time, accents included ("Caffè").
- Small text (dates, addresses, long lines) can get garbled or repeated at Fast. Use
  **Quality**, or cut the text to fewer, shorter lines.

### Styles

Start the prompt with the style and describe it:

- Watercolor illustration of …, soft washes, visible paper texture, loose ink outlines
- Studio Ghibli style anime illustration of …, soft painted backgrounds
- Product photograph of …, plain light grey background, studio lighting, soft shadow
- Flat vector illustration of …, bold colors, simple shapes
- Black and white film photograph of …, grainy, high contrast

### Things that don't help

- **"No X", "without X".** Naming X tends to add it. Describe what *is* there instead.
- **Tag lists** like "masterpiece, best quality, 8k, trending on artstation". They don't add
  quality; they push the picture toward a dramatic stock-photo look. If you want that look, say
  it in words ("dramatic light, high contrast, cinematic").
- **Very small images.** The model is made for about 1 megapixel and up. Tiny sizes (512×512)
  tend to give close-ups and lose the scene.

---

## Editing a photo

Open a photo, then type the change. Kokon keeps the rest of the picture.

### Say the change, name the thing

| Goal | Example |
| --- | --- |
| Swap an object | Replace the yellow armchair with a deep green velvet three-seat sofa |
| Change color | Make the armchair blue |
| Remove | Remove the red scooter |
| Add | Add a small white sailboat on the sea in the bottom left corner |
| Time / weather | Make it a snowy winter night with warm light in the cabin windows |
| Change text | Change the sign to say "PANETTERIA" |
| Expression | Make her laugh with her eyes closed |
| Restyle | Turn it into a watercolor illustration, keep the same scene and composition |
| New background | Change the background to a beach at noon with a few clouds, keep the person the same |

Short edits often work ("green sofa" replaced the armchair and kept the room), but naming the
object removes guesswork: "make it blue" picks the most visible thing in the photo, which may not
be the one you meant.

### Several changes at once

You can ask for more than one change in a sentence. This worked in one go:

> Remove the red scooter, make it a rainy evening with wet reflective cobblestones, turn on the
> café lights, add a ginger cat sitting on a chair, and add an old man with an umbrella walking
> on the right

If one of the changes doesn't happen, do it as a second edit on the result. Each edit is saved as
a new version, so you can always go back.

### "Keep everything else the same"

Kokon already keeps the parts you don't mention. Adding "keep everything else unchanged" or
"keep the person, pose and composition the same" helps when an edit changes too much, for
example when restyling a portrait.

### Vague prompts do nothing

"Make it better", "improve it", "fix it" usually give back almost the same photo. Say what
better means: "brighter and warmer light", "remove the clutter on the table", "sharper, more
contrast, vivid colors".

### Paint an area (region edits)

Brush over the part you want to change, pick an action, and write only about that area:

| Action | What to write | Example |
| --- | --- | --- |
| **Change** | the change for that area | make the jacket red leather |
| **Remove** | nothing, or what to remove | the bin |
| **Replace** | what goes there instead | a ginger cat sitting on the cobblestones |
| **Add** | what to add inside the area | a vase of sunflowers |

Paint a bit larger than the object, including its shadow. Everything outside the brush stays
exactly as it was.

### Combine images (Edit image)

With **Edit image**, image 1 is the photo being changed and the others bring something into it.
Call them **image 1, image 2…** in the order they appear:

> Place the white sneaker from image 2 on the rug in image 1, in front of the armchair, with
> natural shadows matching the room light. Keep the room from image 1 unchanged.

The result takes the shape of image 1. Say which image gives what (the person, the place, the
light, the style).

### A new image from references (New image)

With **New image**, the images you add are references: a person, a product, a pet, a style. The
result is a new picture with its own scene, framing and light, sized by the quality and aspect
ratio you choose (not by image 1). Describe the whole scene and say what to take from each image:

> A photo of the woman from image 1 sitting at a café table by a sunny window, holding the
> coffee cup from image 2 near her face, looking at the camera.

A reference's background stays out unless you ask for it. Prompt+ looks at the references and
writes the details that make them recognizable.

### Expand a photo

Choose how much to extend on each side; the prompt is optional. Describe what should continue
("continue the rocky coastline and the turquoise sea at sunset") if the model guesses wrong.

---

## When the result isn't right

| Problem | Try |
| --- | --- |
| Part of the prompt ignored | Add detail about that part, put it earlier in the prompt |
| Something unwanted appeared | Remove its name from the prompt (even "no …"), describe what should be there |
| Text is wrong | Shorter text in quotes; switch to Quality |
| Close but not quite | Regenerate (new seed) |
| Edit changed too much | Add "keep everything else unchanged", or paint the area and edit only that |
| Edit changed nothing | Be more specific: name the object and the new look |
