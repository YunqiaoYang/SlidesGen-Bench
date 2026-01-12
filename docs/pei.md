
# PEI Technical Specification: The Presentation Editability Intelligence Taxonomy


## 1. Executive Summary
The **Presentation Editability Intelligence (PEI)** framework is a hierarchical standard for evaluating the structural integrity, semantic logic, and editability of AI-generated presentations.

Unlike traditional metrics that measure visual similarity (e.g., FID scores), PEI measures **Editability Depth**. It asserts that a professional presentation is not merely a static image but a complex database of relationships defined by the Office Open XML standard.

### 1.1 The Knockout Rule (Dependency Logic)
The framework operates on a strict **dependency-based knockout mechanism**.
*   Higher levels (e.g., L4 Data) rely on the existence of lower levels (e.g., L3 Structure).
*   **Evaluation Protocol:** If a file fails a specific criterion, evaluation ceases immediately. The file is assigned the highest level it successfully completed.
*   *Example:* A file with perfect animations (L5) but broken charts (L4 failure) is classified as **Level 3**.---

## 2. Input Triage & Routing
The evaluation process begins with **Input Format Analysis**, which determines the evaluation pipeline and the **Maximum Attainable Level (MAL)**.

### Scenario A: The Static Input
*   **Supported Formats:** `.pdf`, `.png`, `.jpg`, `.jpeg`, `.bmp`
*   **Protocol:** **Immediate Termination**.
*   **Maximum Attainable Level:** **L0**
*   **Technical Rationale:** These formats are flattened raster containers. They do not support object separation, text reflow, or XML data binding.

### Scenario B: The Web Input
*   **Supported Formats:** `URL` (Web Viewers, HTML5 Decks, Online Canvas links)
*   **Protocol:** **Visual Inspection Only**.
*   **Maximum Attainable Level:** **L2 (Vector)**
*   **Technical Rationale:** Web viewers render the Document Object Model (DOM) visually but obscure the underlying file structure. It is technically impossible to verify deep editability features—such as Master Slide inheritance (`p:sldMaster`) or embedded Excel binary binding (`c:chart`)—through a standard web interface. Therefore, the score is capped at the limit of visual verification (L2).

### Scenario C: The Native Input
*   **Supported Formats:** `.pptx`, `.potx` (Office Open XML formats)
*   **Protocol:** **Full Deep-Scan Evaluation**.
*   **Maximum Attainable Level:** **L5 (Cinematic)**
*   **Technical Rationale:** These files provide full access to the XML schema, allowing verification of Master Slides, Data Relationships, and Animation Timings.
---

## 3. The PEI Hierarchy: Detailed Definitions
This section defines the technical criteria for each level.
![alt text](../images/pipeline.png)
### Phase 0: The Flat Phase
#### **Level 0: Static (The Flat Image)** 🖼️
*   **Definition:** The content is indistinguishable from a static bitmap.
*   **Technical Hallmark:** Content is flattened. Text is rasterized pixels, not character strings.
*   **Critical Failure Condition:** Text is unselectable; individual elements cannot be isolated.

<figure>
  <img src="../images/image-1.png" alt="L0 NotebookLM as an uneditable PDF example">
  <figcaption>Level 0: Static (Uneditable PDF)</figcaption>
</figure>

### Phase 1: The Visual Phase (Surface Fidelity)
#### **Level 1: Patchwork (The Static Toy)** 🧩
*   **Definition:** The file contains editable elements, but they are fragmented and structurally broken.
*   **Technical Hallmark:** "OCR-style" reconstruction. Paragraphs are split into multiple single-line text boxes. Layouts rely on absolute positioning coordinates rather than flow.
*   **Critical Failure Condition (Knockout):**
    *   **No Text Reflow:** Editing text causes it to overlap other elements instead of wrapping.
    *   **Fragmentation:** A 5-line paragraph is represented by 5 separate text objects.
<figure>
  <img src="../images/image-3.png" alt="L0 NotebookLM as an uneditable PDF example">
  <img src="../images/image-2.png" alt="L0 NotebookLM as an uneditable PDF example">
  <figcaption>Level 1: Patchwork (OCR text and a broken backgroud)</figcaption>
</figure>

#### **Level 2: Vector (The Visual Toy)** 🎨
*   **Definition:** Visual clarity is achieved via vector graphics, but elements lack logical grouping.
*   **Technical Hallmark:** Usage of SVG paths and Scalable Primitives. Graphics remain sharp at 400% zoom.
*   **Critical Failure Condition (Knockout):**
    *   **Visual Mimicry:** A chart is drawn using rectangles and lines, not a data object.
    *   **Isolation:** Complex graphics (e.g., an icon + label) are not Grouped, requiring users to move them individually.
    *   *(Note: This is the hard ceiling for URL inputs).*
<figure>
  <img src="../images/image-4.png" alt="L0 NotebookLM as an uneditable PDF example">
  <img src="../images/image-5.png" alt="L0 NotebookLM as an uneditable PDF example">
  <figcaption>Level 2: Vector (SVG Path and Scalable Primitives)</figcaption>
</figure>

### Phase 2: The Structural Phase (Logic & Data)
#### **Level 3: Structural (The Functional Tool)** 🏗️
*   **Definition:** The system adheres to presentation software logic (Masters and Grouping).
*   **Technical Hallmark:**
    *   **Logical Grouping:** Related vector elements are bound using the Group function.
    *   **Master Inheritance:** The file utilizes the `<p:sldMaster>` schema. Layout changes in the Master View propagate globally to all slides.
*   **Critical Failure Condition (Knockout):**
    *   **Hardcoded Layouts:** Every slide background is drawn independently; the Master Slide is blank or unused.
<figure>
  <img src="../images/image-6.png" alt="L0 NotebookLM as an uneditable PDF example">
  <figcaption>Origin slides(show the use of grouping)</figcaption>
  <img src="../images/image-7.png" alt="L0 NotebookLM as an uneditable PDF example">
  <figcaption>Master of the origin slides</figcaption>
  <figcaption>Level 3: Strucural (Use Master and Grouping)</figcaption>
</figure>

#### **Level 4: Parametric (The Enterprise Tool)** 📈
*   **Definition:** Visuals are driven by native data parameters.
*   **Technical Hallmark:**
    *   **Native Data Binding:** Charts are instantiated as `<c:chart>` objects linked to an embedded Excel binary (`.xlsx`).
    *   **SmartArt/Diagrams:** Process flows use semantic connectors, not just distinct lines.
*   **Critical Failure Condition (Knockout):**
    *   Charts are "drawn" (Vector shapes) instead of "generated" (Data objects).
    *   "Edit Data" workflow is broken or missing.### Phase 3: The Experience Phase (Time & Narrative)
<figure>
  <img src="../images/image-8.png" alt="L0 NotebookLM as an uneditable PDF example">
  <figcaption>Level 4:Parametric  (Native data in the chart )(Human-made)</figcaption>
</figure>

#### **Level 5: Cinematic (The Dynamic Experience)** 🎬
*   **Definition:** The presentation functions as a directed, temporal narrative.
*   **Technical Hallmark:**
    *   **Animation Logic:** Elements utilize Build-In/Build-Out effects sequences that match the reading order.
    *   **Native Media:** Video/Audio is embedded in the DOM with playback controls.
    *   **Critical Failure Condition (Knockout):**
    *   Static slides (no transitions).
    *   Videos are merely static screenshots.


## 4. Evaluation Protocols
Select the protocol below matching your input type.
![alt text](../images/image-pro.png)
### Protocol A: The Static Flow 
**Input:** PDF / Image
**Procedure:**
1.  **Format Check:** Identify file extension (`.pdf`, `.png`, etc.).
2.  **Editability Check:** Attempt to select text or move an object.
    *   *Result:* Negative.
3.  **Final Classification:** **Level 0 (Static).**


### Protocol B: The Web Flow
**Input:** URL / Web Viewer
**Constraint:** Max Rating = L2.

**Step 1: Text Reflow Validation (L1 Check)**
*   *Action:* Click on a text block in the web view.
*   *Check:* Is it selectable text? If you delete words, does the text box resize or reflow naturally?
*   *Decision:*
    *   If Text is Image or Unselectable: **Classify as L0.**
    *   If Text is fragmented/does not reflow: **Classify as L1.**
    *   If Text behaves correctly: **Proceed to Step 2.**

**Step 2: Vector Fidelity Validation (L2 Check)**
*   *Action:* Zoom browser to 400%. Inspect icons and diagrams.
*   *Check:* Are edges crisp (Vector/SVG) or pixelated (Raster)?
*   *Decision:*
    *   If Pixelated: **Classify as L1.**
    *   If Crisp/Vector: **Classify as L2.**

**Step 3: Protocol Termination**
*   *Reasoning:* Web views cannot reliably prove the existence of Master Slides or editable Excel data.
*   *Final Classification:* **Level 2 (Vector).**


### Protocol C: The PPTx Flow
**Input:** PPTX File
**Procedure:** Perform checks sequentially. Stop immediately upon failure.

**Step 1: The Text Integrity Check (L1 Gate)**
*   *Action:* Select a paragraph. Edit the text to double its length.
*   *Criteria:* The text must stay within its container and wrap automatically. The paragraph must be a single object, not multiple lines.
*   *Result:*
    *   **Fail:** Content is uneditable (L0) or fragmented (L1). **STOP.**
    *   **Pass:** **Proceed to Step 2.**

**Step 2: The Vector Graphics Check (L2 Gate)**
*   *Action:* Zoom to 400%. Inspect non-text elements (icons, shapes).
*   *Criteria:* Elements must be vector shapes (Shapes/SVG), not raster screenshots.
*   *Result:*
    *   **Fail (Pixelated):** **Downgrade to Level 1. STOP.**
    *   **Pass:** **Proceed to Step 3.**

**Step 3: The Structural Logic Check (L3 Gate)**
*   *Action A (Grouping):* Click a complex icon. Does it move as one unit (Group) or scatter into pieces?
*   *Action B (Masters):* View -> Slide Master. Add a distinct shape to the layout. Close Master View. Does the shape appear on the slides?
*   *Criteria:* Complex elements must be grouped; Layouts must inherit from Master.
*   *Result:*
    *   **Fail:** **Classify as Level 2. STOP.**
    *   **Pass:** **Proceed to Step 4.**

**Step 4: The Data Native Check (L4 Gate)**
*   *Action:* Identify a chart. Right-click the chart area. Look for "Edit Data."
*   *Criteria:* The "Edit Data" option must exist and successfully open an embedded Excel sheet. Changing a value in Excel must instantly update the chart visual.
*   *Result:*
    *   **Fail (No option/Broken link):** **Classify as Level 3. STOP.**
    *   **Pass:** **Proceed to Step 5.**

**Step 5: The Cinematic Check (L5 Gate)**
*   *Action:* Run "Slide Show" mode from the beginning.
*   *Criteria:* Slides must transition automatically or smoothly. Elements should animate in (Build-ins). Embedded video must play natively.
*   *Result:*
    *   **Fail (Static Show):** **Classify as Level 4. STOP.**
    *   **Pass:** **Classify as Level 5 (Cinematic).**
