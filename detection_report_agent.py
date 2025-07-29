import csv
from swarm import Swarm, Agent
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import os
from textwrap import wrap

client = Swarm()

def read_csv_data(file_path: str):
    entries = []
    with open(file_path, mode='r') as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            row = {key.strip(): value.strip() for key, value in row.items()}
            timestamp = row['Timestamp']
            obj_class = row['Object Class']
            image_path = row.get('Image Path', "")
            if os.path.exists(image_path):
                entries.append((timestamp, obj_class, image_path))
            else:
                print(f"⚠️ Skipped missing image: {image_path}")
    return entries


def generate_descriptions(entries):
    descriptions = []
    for idx, (timestamp, obj_class, _) in enumerate(entries, 1):
        messages = [{
            "role": "user",
            "content": f"""
You are an autonomous crash site analysis assistant.

Below is one detected object from a crash site:

Timestamp: {timestamp}
Object Class: {obj_class}

Generate a structured, investigation-style paragraph that includes the following:
- **Brief Description**: 2–3 technical sentences describing what the object likely is
- **Out of Place**: Yes/No — with 1 sentence explanation
- **Potential Usefulness**: Yes/No — with 1 sentence explanation
- **Photo**: Write: "See attached image"

⚠️ Do NOT include Detection #, timestamp, or object class headings — they are already handled in the PDF layout.
⚠️ Just start the body text directly with “- Brief Description: ...”
"""
        }]

        description_agent = Agent(
            name="Object Description Agent",
            instructions="You generate consistent, crash site object summaries.",
        )

        response = client.run(agent=description_agent, messages=messages)
        description = response.messages[-1]["content"].strip()
        descriptions.append(description)

    return descriptions



def save_text_to_pdf(descriptions: list[str], entries: list[tuple], filename: str):
    c = canvas.Canvas(filename, pagesize=A4)
    width, height = A4
    x_margin = 50
    line_height = 18

    # Title Page
    y = height - 50
    title_font_size = 20
    c.setFont("Helvetica-Bold", title_font_size)
    c.drawCentredString(width / 2, y, "Object Detection Report")
    y -= (title_font_size + 40)

    intro = (
        "This report contains object detection data collected from the sensor. "
        "Each page includes a timestamp, the detected object class, a generated "
        "description, and an associated image."
    )
    c.setFont("Helvetica", 12)
    wrapped_lines = wrap(intro, width=90)
    for line in wrapped_lines:
        c.drawString(x_margin, y, line)
        y -= line_height

    c.showPage()

    for idx, ((timestamp, obj_class, image_path), desc) in enumerate(zip(entries, descriptions), 1):
        y = height - 60

        # Title & metadata
        c.setFont("Helvetica-Bold", 14)
        c.drawString(x_margin, y, f"Detection #{idx}")
        y -= line_height * 2

        c.setFont("Helvetica", 12)
        c.drawString(x_margin, y, f"Timestamp: {timestamp}")
        y -= line_height
        c.drawString(x_margin, y, f"Detected Object: {obj_class}")
        y -= (line_height + 10)

        # Description
        c.setFont("Helvetica-Oblique", 12)
        for line in desc.strip().splitlines():
            for wrapped in wrap(line.strip(), width=95):
                c.drawString(x_margin, y, wrapped)
                y -= line_height

        # Image
        if os.path.exists(image_path):
            try:
                img = ImageReader(image_path)
                img_width = 300
                img_height = 200
                if y - img_height < 60:
                    y -= 60
                c.drawImage(image_path, x_margin, y - img_height, width=img_width, height=img_height)
            except Exception as e:
                print(f"⚠️ Failed to load image {image_path}: {e}")
                c.drawString(x_margin, y, "[Image could not be displayed]")

        c.showPage()

    c.save()
    print(f"\n✅ PDF saved as '{filename}' with {len(entries)} pages.")


def generate_description_from_csv(file_path: str):
    entries = read_csv_data(file_path)

    if not entries:
        print("❌ No valid data found in the CSV.")
        return

    descriptions = generate_descriptions(entries)
    save_text_to_pdf(descriptions, entries, "object_detection_report.pdf")


if __name__ == "__main__":
    generate_description_from_csv("detection_log.csv")
