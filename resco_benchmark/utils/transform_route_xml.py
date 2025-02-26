#!/usr/bin/env python3
import sys
import xml.etree.ElementTree as ET

def transform_xml(input_file, output_file):
    """
    Transforms the XML by adjusting each flow with color="white":
      - Reduces the vehsPerHour of the original flow to 70% of its original value.
      - Inserts two new flows immediately after the original flow.
          * One with type="type1_co2" and one with type="type2_co2".
          * Each new flow gets vehsPerHour equal to 15% of the original.
    All other attributes (begin, departLane, departPos, route, end, and color)
    are preserved, as are the non-white flows.
    """
    tree = ET.parse(input_file)
    root = tree.getroot()

    # Since flows are direct children of <routes>, we can iterate over them.
    # We'll build a new list of children so that we can insert additional flows
    new_children = []
    for child in list(root):
        new_children.append(child)
        if child.tag == "flow" and child.get("color") == "white":
            try:
                orig_count = float(child.get("vehsPerHour"))
            except (TypeError, ValueError):
                print(f"Warning: Could not parse vehsPerHour for flow {child.get('id')}. Skipping transformation.")
                continue

            # Calculate new counts
            new_original_count = orig_count * 0.7
            additional_count = orig_count * 0.15

            # Update the original white flow
            child.set("vehsPerHour", str(new_original_count))

            # Create two new flows based on the original
            # Copy over all attributes from the original
            new_flow1 = ET.Element("flow", child.attrib)
            new_flow2 = ET.Element("flow", child.attrib)

            # Modify the IDs to ensure uniqueness by appending suffixes
            orig_id = child.get("id")
            new_flow1.set("id", orig_id + "_co2_1")
            new_flow2.set("id", orig_id + "_co2_2")

            # Set the new vehicle types
            new_flow1.set("type", "type1_co2")
            new_flow2.set("type", "type2_co2")
            # Set the new vehicle types
            new_flow1.set("color", "cyan")
            new_flow2.set("color", "cyan")

            # Set vehsPerHour for the new flows
            new_flow1.set("vehsPerHour", str(additional_count))
            new_flow2.set("vehsPerHour", str(additional_count))

            # Append the new flows immediately after the original flow
            new_children.append(new_flow1)
            new_children.append(new_flow2)

    # Replace the children of the root with our new list
    root[:] = new_children
    # Pretty-print XML to produce new lines for each element
    ET.indent(tree, space="    ", level=0)
    # Write the modified XML to output_file with an XML declaration and UTF-8 encoding
    tree.write(output_file, encoding="UTF-8", xml_declaration=True)

if __name__ == "__main__":
    # if len(sys.argv) < 3:
    #     print("Usage: python transform_route_xml.py input.xml output.xml")
    #     sys.exit(1)
    input_file = r"C:\Clinical\RESCO\resco_benchmark\environments\2way_single\single-intersection-gen_co2.rou.xml"
    output_file = r"C:\Clinical\RESCO\resco_benchmark\environments\2way_single\single-intersection-gen1_co2.rou.xml"
    transform_xml(input_file, output_file)
    print(f"Transformation complete. Modified file saved as '{output_file}'.")
