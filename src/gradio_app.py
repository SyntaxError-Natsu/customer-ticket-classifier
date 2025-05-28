import gradio as gr
import pandas as pd
import json
from ticket_classifier import TicketClassifier
import warnings
warnings.filterwarnings('ignore')

class GradioApp:
    def __init__(self):
        try:
            self.classifier = TicketClassifier()
            self.app_ready = True
        except Exception as e:
            print(f"Error: {e}")
            self.app_ready = False
        
    def predict_single_ticket(self, ticket_text):
        if not self.app_ready:
            return "Models not loaded. Train models first.", "", "", "Error"
        
        if not ticket_text.strip():
            return "Please enter ticket text.", "", "", "No input"
        
        try:
            result = self.classifier.predict_ticket(ticket_text)
            
            # Ultra-safe prediction display with explicit styling for every element
            prediction_html = f"""
            <div style="background: white; color: black; padding: 20px; border-radius: 8px; border: 2px solid #007bff; margin: 10px 0; font-family: Arial, sans-serif;">
                <h3 style="color: #007bff; margin-bottom: 20px; font-weight: bold; font-size: 20px;">🎯 Prediction Results</h3>
                
                <div style="background: #f8f9fa; color: black; padding: 15px; margin: 10px 0; border-radius: 6px; border-left: 4px solid #007bff;">
                    <h4 style="color: #007bff; margin: 0 0 10px 0; font-weight: 600; font-size: 16px;">📋 Issue Type</h4>
                    <p style="margin: 5px 0; font-size: 16px; color: black; font-weight: bold;">{result['predicted_issue_type']}</p>
                    <p style="margin: 0; color: #666; font-size: 14px;">Confidence: {result['issue_confidence']:.1%}</p>
                </div>
                
                <div style="background: #f8f9fa; color: black; padding: 15px; margin: 10px 0; border-radius: 6px; border-left: 4px solid #28a745;">
                    <h4 style="color: #28a745; margin: 0 0 10px 0; font-weight: 600; font-size: 16px;">⚡ Urgency Level</h4>
                    <p style="margin: 5px 0; font-size: 16px; color: black; font-weight: bold;">{result['predicted_urgency_level']}</p>
                    <p style="margin: 0; color: #666; font-size: 14px;">Confidence: {result['urgency_confidence']:.1%}</p>
                </div>
            </div>
            """
            
            # Ultra-safe entities display
            entities = result['extracted_entities']
            entities_html = f"""
            <div style="background: white; color: black; padding: 20px; border-radius: 8px; border: 2px solid #28a745; margin: 10px 0; font-family: Arial, sans-serif;">
                <h3 style="color: #28a745; margin-bottom: 20px; font-weight: bold; font-size: 22px;">🏷️ Extracted Entities</h3>
            """
            
            if entities.get('products'):
                entities_html += f"""
                <div style="background: #f8f9fa; color: black; padding: 15px; margin: 10px 0; border-radius: 6px; border-left: 4px solid #007bff;">
                    <h4 style="color: #007bff; margin: 0 0 10px 0; font-weight: 600; font-size: 16px;">🔧 Products</h4>
                    <p style="margin: 0; font-size: 16px; color: black; font-weight: bold;">{', '.join(entities['products'])}</p>
                </div>
                """
            
            if entities.get('dates'):
                entities_html += f"""
                <div style="background: #f8f9fa; color: black; padding: 15px; margin: 10px 0; border-radius: 6px; border-left: 4px solid #28a745;">
                    <h4 style="color: #28a745; margin: 0 0 10px 0; font-weight: 600; font-size: 16px;">📅 Dates</h4>
                    <p style="margin: 0; font-size: 16px; color: black; font-weight: bold;">{', '.join(entities['dates'])}</p>
                </div>
                """
            
            if entities.get('durations'):
                entities_html += f"""
                <div style="background: #f8f9fa; color: black; padding: 15px; margin: 10px 0; border-radius: 6px; border-left: 4px solid #17a2b8;">
                    <h4 style="color: #17a2b8; margin: 0 0 10px 0; font-weight: 600; font-size: 16px;">⏱️ Durations</h4>
                    <p style="margin: 0; font-size: 16px; color: black; font-weight: bold;">{', '.join(entities['durations'])}</p>
                </div>
                """
            
            if entities.get('order_numbers'):
                entities_html += f"""
                <div style="background: #f8f9fa; color: black; padding: 15px; margin: 10px 0; border-radius: 6px; border-left: 4px solid #6f42c1;">
                    <h4 style="color: #6f42c1; margin: 0 0 10px 0; font-weight: 600; font-size: 16px;">📋 Order Numbers</h4>
                    <p style="margin: 0; font-size: 16px; color: black; font-weight: bold;">{', '.join(entities['order_numbers'])}</p>
                </div>
                """
            
            if entities.get('complaint_keywords'):
                entities_html += f"""
                <div style="background: #f8f9fa; color: black; padding: 15px; margin: 10px 0; border-radius: 6px; border-left: 4px solid #ffc107;">
                    <h4 style="color: #e0a800; margin: 0 0 10px 0; font-weight: 600; font-size: 16px;">⚠️ Problem Keywords</h4>
                    <p style="margin: 0; font-size: 16px; color: black; font-weight: bold;">{', '.join(entities['complaint_keywords'])}</p>
                </div>
                """
            
            if entities.get('urgency_indicators'):
                entities_html += f"""
                <div style="background: #f8f9fa; color: black; padding: 15px; margin: 10px 0; border-radius: 6px; border-left: 4px solid #dc3545;">
                    <h4 style="color: #dc3545; margin: 0 0 10px 0; font-weight: 600; font-size: 16px;">🚨 Urgency Indicators</h4>
                    <p style="margin: 0; font-size: 16px; color: black; font-weight: bold;">{', '.join(entities['urgency_indicators'])}</p>
                </div>
                """
            
            if entities.get('contacted_support'):
                entities_html += f"""
                <div style="background: #f8f9fa; color: black; padding: 15px; margin: 10px 0; border-radius: 6px; border-left: 4px solid #6c757d;">
                    <h4 style="color: #6c757d; margin: 0 0 10px 0; font-weight: 600; font-size: 16px;">📞 Support Contact</h4>
                    <p style="margin: 0; font-size: 16px; color: black; font-weight: bold;">Previous contact detected</p>
                </div>
                """
            
            entities_html += "</div>"
            
            if not any([entities.get('products'), entities.get('dates'), entities.get('durations'),
                       entities.get('order_numbers'), entities.get('complaint_keywords'), 
                       entities.get('urgency_indicators'), entities.get('contacted_support')]):
                entities_html = """
                <div style="background: #fff3cd; color: #856404; padding: 20px; border-radius: 8px; border: 2px solid #ffeaa7; font-family: Arial, sans-serif;">
                    <h3 style="color: #856404; text-align: center; font-weight: bold; font-size: 20px;">🏷️ No Entities Found</h3>
                    <p style="color: #856404; text-align: center; font-size: 16px;">No specific products, dates, or order numbers detected.</p>
                </div>
                """
            
            json_output = json.dumps(result, indent=2)
            return prediction_html, entities_html, json_output, "✅ Analysis complete"
            
        except Exception as e:
            return f"Error: {str(e)}", "", "", "❌ Failed"
    
    def process_batch_tickets(self, file):
        if not self.app_ready:
            return "Models not loaded.", "", "Error"
        
        try:
            if file.name.endswith('.csv'):
                df = pd.read_csv(file.name)
            elif file.name.endswith('.xlsx'):
                df = pd.read_excel(file.name)
            else:
                return "Upload CSV or Excel file.", "", "Invalid format"
            
            if 'ticket_text' not in df.columns:
                return "File must have 'ticket_text' column.", "", "Missing column"
            
            results = []
            for idx, row in df.iterrows():
                if idx >= 50:
                    break
                    
                result = self.classifier.predict_ticket(row['ticket_text'])
                results.append({
                    'ticket_id': idx + 1,
                    'text_preview': str(row['ticket_text'])[:80] + "...",
                    'issue_type': result['predicted_issue_type'],
                    'urgency': result['predicted_urgency_level'],
                    'issue_confidence': f"{result['issue_confidence']:.1%}",
                    'urgency_confidence': f"{result['urgency_confidence']:.1%}"
                })
            
            results_df = pd.DataFrame(results)
            
            # Ultra-safe batch summary with explicit styling
            summary_html = f"""
            <div style="background: white; color: black; padding: 20px; border-radius: 8px; border: 2px solid #17a2b8; font-family: Arial, sans-serif;">
                <h3 style="color: #17a2b8; font-weight: bold; font-size: 20px; margin-bottom: 15px;">📊 Batch Results</h3>
                <p style="color: black; font-size: 16px; font-weight: bold; margin-bottom: 15px;">Processed: {len(results_df)} tickets</p>
                
                <h4 style="color: #007bff; font-size: 16px; margin: 15px 0 10px 0;">Issue Types:</h4>
                <div style="background: #f8f9fa; color: black; padding: 10px; border-radius: 4px; border: 1px solid #dee2e6; font-family: monospace; white-space: pre-wrap; font-size: 14px;">{results_df['issue_type'].value_counts().to_string()}</div>
                
                <h4 style="color: #28a745; font-size: 16px; margin: 15px 0 10px 0;">Urgency Levels:</h4>
                <div style="background: #f8f9fa; color: black; padding: 10px; border-radius: 4px; border: 1px solid #dee2e6; font-family: monospace; white-space: pre-wrap; font-size: 14px;">{results_df['urgency'].value_counts().to_string()}</div>
            </div>
            """
            
            return summary_html, results_df, "✅ Batch complete"
            
        except Exception as e:
            return f"Error: {str(e)}", "", "❌ Failed"
    
    def create_interface(self):
        # Minimal CSS
        custom_css = """
        .gradio-container {
            max-width: 1200px;
            margin: 0 auto;
            font-family: Arial, sans-serif;
        }
        
        .gr-button-primary {
            background-color: #007bff !important;
            border-color: #007bff !important;
            color: white !important;
            font-weight: 600 !important;
            border-radius: 6px !important;
        }
        
        .gr-button-primary:hover {
            background-color: #0056b3 !important;
            border-color: #0056b3 !important;
        }
        """
        
        with gr.Blocks(title="Ticket Classifier", css=custom_css) as interface:
            
            # Ultra-safe header
            gr.HTML("""
            <div style="background: white; color: black; padding: 30px; text-align: center; border-bottom: 3px solid #007bff; font-family: Arial, sans-serif;">
                <h1 style="color: #2c3e50; margin: 0; font-weight: 700; font-size: 28px;">🎫 Support Ticket Classifier</h1>
                <p style="color: #6c757d; margin: 10px 0 0 0; font-size: 16px;">Classify tickets and extract entities</p>
            </div>
            """)
            
            with gr.Tab("Single Ticket"):
                with gr.Row(equal_height=True):
                    with gr.Column(scale=1):
                        ticket_input = gr.Textbox(
                            label="Ticket Description",
                            placeholder="Enter customer issue here...",
                            lines=8
                        )
                        
                        predict_btn = gr.Button("🔮 Analyze", variant="primary", size="lg")
                        status_output = gr.Textbox(label="Status", interactive=False)
                    
                    with gr.Column(scale=1):
                        prediction_output = gr.HTML(label="Predictions")
                        entities_output = gr.HTML(label="Entities")
                
                with gr.Accordion("JSON Output", open=False):
                    json_output = gr.Code(label="Raw Result", language="json")
                
                # Ultra-safe examples
                gr.HTML("""
                <div style="background: white; color: black; padding: 15px; border-radius: 8px; margin: 20px 0; border: 2px solid #6f42c1; font-family: Arial, sans-serif;">
                    <h3 style='text-align: center; margin: 0; color: #6f42c1; font-size: 18px;'>📝 Try These Examples:</h3>
                </div>
                """)
                
                examples = [
                    "My SmartWatch V2 stopped working after 3 days. Won't turn on at all.",
                    "Payment issue for order #29224. Was charged twice for my Vision LED TV.",
                    "URGENT: Can't log into my account. Need help immediately!",
                    "Installation failed for RoboChef Blender. Setup guide unclear.",
                    "Order #30903 is 2 weeks late. Very frustrated with delay."
                ]
                
                gr.Examples(
                    examples=examples,
                    inputs=ticket_input,
                    outputs=[prediction_output, entities_output, json_output, status_output],
                    fn=self.predict_single_ticket
                )
            
            with gr.Tab("Batch Processing"):
                gr.HTML("""
                <div style="background: white; color: black; padding: 15px; border-radius: 8px; margin: 10px 0; border: 2px solid #28a745; font-family: Arial, sans-serif;">
                    <h3 style="color: #28a745; text-align: center; margin: 0; font-size: 18px;">📁 Process Multiple Tickets</h3>
                    <p style="color: black; text-align: center; margin: 10px 0 0 0; font-size: 14px;">Upload CSV/Excel with 'ticket_text' column</p>
                </div>
                """)
                
                with gr.Row():
                    with gr.Column():
                        file_input = gr.File(label="Upload File", file_types=[".csv", ".xlsx"])
                        process_btn = gr.Button("🚀 Process", variant="primary", size="lg")
                        batch_status = gr.Textbox(label="Status", interactive=False)
                
                with gr.Row():
                    with gr.Column():
                        summary_output = gr.HTML(label="Summary")
                    with gr.Column():
                        results_output = gr.Dataframe(label="Results")
            
            with gr.Tab("About"):
                gr.HTML("""
                <div style="background: white; color: black; padding: 25px; border-radius: 8px; border: 2px solid #17a2b8; font-family: Arial, sans-serif;">
                    <h2 style="color: #2c3e50; text-align: center; font-weight: bold; font-size: 24px;">🤖 About</h2>
                    
                    <div style="background: #f8f9fa; color: black; padding: 15px; margin: 15px 0; border-radius: 6px; border-left: 4px solid #007bff;">
                        <h3 style="color: #007bff; margin: 0 0 10px 0; font-size: 18px;">🎯 What it does:</h3>
                        <ul style="color: black; margin: 0; line-height: 1.6; font-size: 14px;">
                            <li style="color: black;">Classifies tickets into issue types (Billing, Installation, etc.)</li>
                            <li style="color: black;">Predicts urgency levels (Low, Medium, High)</li>
                            <li style="color: black;">Extracts entities (products, dates, order numbers)</li>
                        </ul>
                    </div>
                    
                    <div style="background: #f8f9fa; color: black; padding: 15px; margin: 15px 0; border-radius: 6px; border-left: 4px solid #28a745;">
                        <h3 style="color: #28a745; margin: 0 0 10px 0; font-size: 18px;">🛠️ How it works:</h3>
                        <ul style="color: black; margin: 0; line-height: 1.6; font-size: 14px;">
                            <li style="color: black;">Uses Random Forest and Logistic Regression models</li>
                            <li style="color: black;">TF-IDF features + engineered text features</li>
                            <li style="color: black;">Rule-based entity extraction with regex patterns</li>
                        </ul>
                    </div>
                    
                    <div style="background: #f8f9fa; color: black; padding: 15px; margin: 15px 0; border-radius: 6px; border-left: 4px solid #17a2b8;">
                        <h3 style="color: #17a2b8; margin: 0 0 10px 0; font-size: 18px;">📊 Performance:</h3>
                        <ul style="color: black; margin: 0; line-height: 1.6; font-size: 14px;">
                            <li style="color: black;">Issue classification: 60-80% accuracy</li>
                            <li style="color: black;">Urgency prediction: 55-75% accuracy</li>
                            <li style="color: black;">Entity extraction: High precision for structured data</li>
                        </ul>
                    </div>
                </div>
                """)
            
            # Connect functions
            predict_btn.click(
                fn=self.predict_single_ticket,
                inputs=ticket_input,
                outputs=[prediction_output, entities_output, json_output, status_output]
            )
            
            process_btn.click(
                fn=self.process_batch_tickets,
                inputs=file_input,
                outputs=[summary_output, results_output, batch_status]
            )
        
        return interface

def main():
    print("🚀 Starting Ticket Classifier...")
    app = GradioApp()
    
    if not app.app_ready:
        print("❌ Train models first using the Jupyter notebook")
        return
    
    interface = app.create_interface()
    print("✅ Interface ready!")
    interface.launch(share=True)

if __name__ == "__main__":
    main()
