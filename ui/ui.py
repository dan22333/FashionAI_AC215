import gradio as gr


def recommend_fashion(image, event_type, fashion_item):
    if event_type.lower() == "wedding":
        return f"For the {fashion_item}, we recommend something elegant, preferably in neutral or classic colors."
    else:
        return f"For the {fashion_item}, we suggest something comfortable and stylish."


def gradio_fashion_interface():
    # Inputs (using the updated gr.components with correct type)
    image_input = gr.Image(label="Upload your fashion item image (e.g., pants)", type="filepath")
    event_input = gr.Textbox(lines=1, placeholder="Event type (e.g., Wedding, Casual, Party)", label="Event Type")
    fashion_item_input = gr.Textbox(lines=1, placeholder="Item to match (e.g., shirt, shoes)", label="Fashion Item to Match")
    

    recommendation_output = gr.Textbox(label="Recommended Fashion")

    # Interface Layout
    interface = gr.Interface(
        fn=recommend_fashion,
        inputs=[image_input, event_input, fashion_item_input],
        outputs=recommendation_output,
        title="Fashion Recommendation System",
        description="Upload an image of a fashion item, describe your event, and get fashion advice!",
        examples=[["example_image.png", "Wedding", "Shirt"]]
    )

    interface.launch(share=True)

if __name__ == "__main__":
    gradio_fashion_interface()
