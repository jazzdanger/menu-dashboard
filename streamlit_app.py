import streamlit as st
import cv2
import smtplib
from email.message import EmailMessage
import os
import pywhatkit as kit
import time
import speech_recognition as sr
from googlesearch import search
import requests
from bs4 import BeautifulSoup
from twilio.rest import Client
from twilio.base.exceptions import TwilioRestException
import psutil
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import tempfile


SENDER_EMAIL = os.environ.get("GMAIL_USERNAME")
SENDER_PASSWORD = os.environ.get("GMAIL_PASSWORD")
ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID")
AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.environ.get("TWILIO_PHONE_NUMBER")
TWILIO_WHATSAPP_NUMBER = "whatsapp:+14155238886" # Twilio's Sandbox number

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# --- Helper Functions from your script (with minor adaptations for Streamlit) ---

def send_email(to_email, subject, body):
    """Sends an email using Gmail's SMTP server."""
    if not SENDER_EMAIL or not SENDER_PASSWORD:
        st.error("Email credentials (GMAIL_USERNAME, GMAIL_PASSWORD) are not set as environment variables.")
        return

    msg = EmailMessage()
    msg['From'] = SENDER_EMAIL
    msg['To'] = to_email
    msg['Subject'] = subject
    msg.set_content(body)

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(SENDER_EMAIL, SENDER_PASSWORD)
            smtp.send_message(msg)
        st.success("Email sent successfully!")
    except smtplib.SMTPAuthenticationError:
        st.error("SMTP Authentication Error. Check your email credentials or App Password.")
    except Exception as e:
        st.error(f"Error sending email: {e}")

def audio_to_text(audio_file):
    """Transcribes an uploaded audio file to text."""
    if audio_file is None:
        st.warning("Please upload an audio file.")
        return None
    
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data)
            return text
        except sr.UnknownValueError:
            st.error("Google Speech Recognition could not understand the audio.")
            return None
        except sr.RequestError as e:
            st.error(f"Could not request results from Google Speech Recognition service; {e}")
            return None

def google_search_with_snippets(query, num_results=5):
    """Performs a Google search and fetches snippets from the results."""
    results = []
    try:
        urls = list(search(query, num_results=num_results, lang="en"))
        for u in urls:
            try:
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0 Safari/537.36"
                }
                r = requests.get(u, timeout=8, headers=headers)
                if r.status_code == 200:
                    soup = BeautifulSoup(r.text, "html.parser")
                    title = soup.title.string.strip() if soup.title else "No title"
                    paragraphs = " ".join(p.get_text(strip=True) for p in soup.find_all("p"))
                    snippet = " ".join(paragraphs.split()[:30]) + "..."
                    results.append({"title": title, "snippet": snippet, "link": u})
                else:
                    results.append({"title": "Could not fetch", "snippet": f"Status code: {r.status_code}", "link": u})
            except Exception as e:
                results.append({"title": "Error fetching page", "snippet": str(e), "link": u})
            time.sleep(0.5) # Be respectful to servers
    except Exception as e:
        st.error(f"An error occurred during Google Search: {e}")
    return results


def swap_faces(img1_bytes, img2_bytes):
    """Swaps faces between two images provided as bytes."""
    if img1_bytes is None or img2_bytes is None:
        st.warning("Please upload both images to swap faces.")
        return None

    # Convert image bytes to OpenCV format
    img1 = cv2.imdecode(np.frombuffer(img1_bytes.read(), np.uint8), 1)
    img2 = cv2.imdecode(np.frombuffer(img2_bytes.read(), np.uint8), 1)

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    faces1 = face_cascade.detectMultiScale(gray1, 1.1, 4)
    faces2 = face_cascade.detectMultiScale(gray2, 1.1, 4)

    if len(faces1) == 0 or len(faces2) == 0:
        st.error("Face not detected in one or both images. Please try with different images.")
        return None

    (x1, y1, w1, h1) = faces1[0]
    (x2, y2, w2, h2) = faces2[0]

    face1_roi = img1[y1:y1 + h1, x1:x1 + w1]
    face2_roi = img2[y2:y2 + h2, x2:x2 + w2]

    face1_resized = cv2.resize(face1_roi, (w2, h2))
    face2_resized = cv2.resize(face2_roi, (w1, h1))

    swap_img1 = img1.copy()
    swap_img2 = img2.copy()

    swap_img1[y1:y1 + h1, x1:x1 + w1] = face2_resized
    swap_img2[y2:y2 + h2, x2:x2 + w2] = face1_resized

    return swap_img1, swap_img2


def create_digital_image():
    """Creates a simple digital image with shapes and text."""
    width, height = 800, 600
    img = Image.new('RGB', (width, height), color=(240, 240, 255))
    draw = ImageDraw.Draw(img)

    # Draw shapes
    draw.rectangle([50, 50, 250, 150], fill=(255, 87, 51), outline='black', width=3)
    draw.ellipse([300, 200, 500, 400], fill=(76, 175, 80), outline='black', width=3)
    draw.line([50, 500, 750, 500], fill=(33, 150, 243), width=10)

    # Draw text
    try:
        font = ImageFont.truetype("arial.ttf", 40)
    except IOError:
        font = ImageFont.load_default() # Fallback font
    draw.text((300, 75), "Hello, Streamlit!", fill=(50, 50, 50), font=font)
    return img


def check_memory_usage():
    """Returns a dictionary with system memory usage."""
    mem = psutil.virtual_memory()
    return {
        "Total": f"{mem.total / (1024**3):.2f} GB",
        "Available": f"{mem.available / (1024**3):.2f} GB",
        "Used": f"{mem.used / (1024**3):.2f} GB",
        "Percentage": f"{mem.percent}%"
    }

def website_data(url):
    """Fetches and returns the text content of a static website."""
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            return soup.get_text(separator="\n", strip=True)
        else:
            return f"Error: Failed to access site, status code: {response.status_code}"
    except Exception as e:
        return f"Error: An unexpected error occurred: {e}"

def send_sms(recipient, message_body):
    """Sends an SMS using Twilio."""
    if not ACCOUNT_SID or not AUTH_TOKEN or not TWILIO_PHONE_NUMBER:
        st.error("Twilio credentials are not set as environment variables.")
        return
    try:
        client = Client(ACCOUNT_SID, AUTH_TOKEN)
        message = client.messages.create(
            body=message_body,
            from_=TWILIO_PHONE_NUMBER,
            to=recipient
        )
        st.success(f"SMS sent successfully! SID: {message.sid}")
    except TwilioRestException as e:
        st.error(f"Error sending SMS: {e}")

def send_twilio_whatsapp(recipient, message_body):
    """Sends a WhatsApp message using Twilio."""
    if not ACCOUNT_SID or not AUTH_TOKEN:
        st.error("Twilio credentials (ACCOUNT_SID, AUTH_TOKEN) are not set as environment variables.")
        return
    try:
        client = Client(ACCOUNT_SID, AUTH_TOKEN)
        message = client.messages.create(
            from_=TWILIO_WHATSAPP_NUMBER,
            body=message_body,
            to=f'whatsapp:{recipient}'
        )
        st.success(f"WhatsApp message sent successfully! SID: {message.sid}")
    except TwilioRestException as e:
        st.error(f"Error sending WhatsApp message via Twilio: {e}")

def make_call(recipient, message_to_say):
    """Makes a call using Twilio."""
    if not ACCOUNT_SID or not AUTH_TOKEN or not TWILIO_PHONE_NUMBER:
        st.error("Twilio credentials are not set as environment variables.")
        return
    try:
        client = Client(ACCOUNT_SID, AUTH_TOKEN)
        twiml = f"<Response><Say voice='alice'>{message_to_say}</Say></Response>"
        call = client.calls.create(
            twiml=twiml,
            to=recipient,
            from_=TWILIO_PHONE_NUMBER
        )
        st.success(f"Call initiated! SID: {call.sid}")
    except TwilioRestException as e:
        st.error(f"Twilio call error: {e}")


# --- Streamlit App UI ---

st.set_page_config(page_title="Multi-Tool Python Suite", layout="wide")

st.title("🐍 Multi-Tool Python Suite")
st.write("A Streamlit interface for various Python automation and utility scripts.")

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
menu_choice = st.sidebar.radio("Go to:", [
    "Communication", "Image & Video", "Audio", "Google Search", "System Utilities"
])

# --- Communication Menu ---
if menu_choice == "Communication":
    st.header("Communication Tools")
    comm_option = st.selectbox("Choose a communication method:", [
        "Send a WhatsApp message", "Send an email", "Send an SMS message", "Send a WhatsApp message without your number", "Send a call"
    ])

    if comm_option == "Send an email":
        st.subheader("📧 Send an Email")
        to_email = st.text_input("Recipient Email:")
        subject = st.text_input("Subject:")
        body = st.text_area("Email Body:")
        if st.button("Send Email"):
            if to_email and subject and body:
                send_email(to_email, subject, body)
            else:
                st.warning("Please fill in all fields.")

    elif comm_option == "Send a WhatsApp message":
        st.subheader("📱 Send a WhatsApp Message (via Web)")
        st.info("This uses `pywhatkit` and will open WhatsApp Web in a new browser tab to send the message.")
        number = st.text_input("Recipient's Number (e.g., +91XXXXXXXXXX):")
        message = st.text_area("Message:")
        if st.button("Send Instant Message"):
            if number and message:
                try:
                    kit.sendwhatmsg_instantly(number, message, wait_time=15, tab_close=True)
                    st.success("WhatsApp Web should have opened. The message will be sent shortly.")
                except Exception as e:
                    st.error(f"Error with pywhatkit: {e}")
            else:
                st.warning("Please provide a number and a message.")
    
    elif comm_option == "Send a WhatsApp message without your number":
        st.subheader("📱 Send a WhatsApp Message (via Twilio)")
        st.info("This uses the Twilio API to send a WhatsApp message. The recipient must have first sent a message to the Twilio sandbox number.")
        recipient = st.text_input("Recipient WhatsApp Number (e.g., +91XXXXXXXXXX):")
        body = st.text_area("Message:")
        if st.button("Send Twilio WhatsApp"):
            if recipient and body:
                send_twilio_whatsapp(recipient, body)
            else:
                st.warning("Please provide a recipient number and a message.")

    elif comm_option == "Send an SMS message":
        st.subheader("💬 Send an SMS")
        st.info("This uses Twilio to send an SMS.")
        recipient = st.text_input("Recipient Phone Number (e.g., +14155552671):")
        body = st.text_area("Message:")
        if st.button("Send SMS"):
            if recipient and body:
                send_sms(recipient, body)
            else:
                st.warning("Please provide a recipient number and a message.")

    elif comm_option == "Send a call":
        st.subheader("📞 Make a Call")
        st.info("This uses Twilio to make an automated call.")
        recipient = st.text_input("Recipient Phone Number to Call (e.g., +14155552671):")
        message_to_say = st.text_area("Message to be spoken in the call:")
        if st.button("Initiate Call"):
            if recipient and message_to_say:
                make_call(recipient, message_to_say)
            else:
                st.warning("Please provide a recipient number and a message.")


# --- Image & Video Menu ---
elif menu_choice == "Image & Video":
    st.header("Image & Video Tools")
    img_option = st.selectbox("Choose a tool:", [
        "Capture an image", "Record a video", "create a digital image", "swap faces of 2 images"
    ])

    if img_option == "Capture an image":
        st.subheader("📸 Capture Image from Webcam")
        img_file_buffer = st.camera_input("Press the button below to capture an image")
        if img_file_buffer is not None:
            st.success("Image captured!")
            st.image(img_file_buffer, caption="Your Captured Image")
            bytes_data = img_file_buffer.getvalue()
            cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            if len(faces) > 0:
                st.info(f"Detected {len(faces)} face(s) in the image.")
                for (x, y, w, h) in faces:
                    cv2.rectangle(cv2_img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                st.image(cv2_img, channels="BGR", caption="Image with Face Detection")
    
    elif img_option == "Record a video":
        st.subheader("📹 Record a Video from Webcam")
        
        run = st.checkbox('Start/Stop Recording')
        FRAME_WINDOW = st.image([])
        
        if 'video_filename' not in st.session_state:
            st.session_state.video_filename = None

        if run:
            st.info("Recording started... Uncheck the box to stop.")
            cap = cv2.VideoCapture(0)
            
            # Define the codec and create VideoWriter object
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Or 'XVID'
            
            # Create a temporary file to save the video
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmpfile:
                st.session_state.video_filename = tmpfile.name
                out = cv2.VideoWriter(st.session_state.video_filename, fourcc, 20.0, (width, height))

                while run:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Failed to capture frame.")
                        break
                    out.write(frame)
                    FRAME_WINDOW.image(frame, channels="BGR")
                
                cap.release()
                out.release()
                st.success(f"Recording stopped.")
        
        if st.session_state.video_filename and not run:
            st.info("Video playback:")
            video_file = open(st.session_state.video_filename, 'rb')
            video_bytes = video_file.read()
            st.video(video_bytes)
            
            with open(st.session_state.video_filename, "rb") as file:
                st.download_button(
                    label="Download Video",
                    data=file,
                    file_name="recorded_video.mp4",
                    mime="video/mp4"
                )


    elif img_option == "swap faces of 2 images":
        st.subheader("🎭 Face Swap")
        col1, col2 = st.columns(2)
        with col1:
            img1_buffer = st.file_uploader("Upload First Image", type=['png', 'jpg', 'jpeg'])
            if img1_buffer:
                st.image(img1_buffer, caption="First Image")
        with col2:
            img2_buffer = st.file_uploader("Upload Second Image", type=['png', 'jpg', 'jpeg'])
            if img2_buffer:
                st.image(img2_buffer, caption="Second Image")

        if st.button("Swap Faces"):
            if img1_buffer and img2_buffer:
                swapped_images = swap_faces(img1_buffer, img2_buffer)
                if swapped_images:
                    swapped1, swapped2 = swapped_images
                    st.success("Faces swapped successfully!")
                    col3, col4 = st.columns(2)
                    with col3:
                        st.image(swapped1, channels="BGR", caption="Swapped Image 1")
                    with col4:
                        st.image(swapped2, channels="BGR", caption="Swapped Image 2")
            else:
                st.warning("Please upload both images.")

    elif img_option == "create a digital image":
        st.subheader("🎨 Create a Digital Image")
        if st.button("Generate Image"):
            generated_image = create_digital_image()
            st.image(generated_image, caption="Generated Digital Art")

# --- Audio Menu ---
elif menu_choice == "Audio":
    st.header("Audio Tools")
    st.subheader("🎤 Audio to Text Transcription")
    audio_file = st.file_uploader("Upload an audio file (.wav, .flac)", type=['wav', 'flac'])
    
    if audio_file is not None:
        st.audio(audio_file)
        if st.button("Transcribe Audio"):
            with st.spinner("Transcribing..."):
                transcribed_text = audio_to_text(audio_file)
            if transcribed_text:
                st.success("Transcription successful!")
                st.text_area("Transcribed Text", transcribed_text, height=200)

# --- Google Search Menu ---
elif menu_choice == "Google Search":
    st.header("🔍 Google Search")

    query = st.text_input("Enter your search query:")

    if st.button("Search"):
        if query.strip():  # Check if the query is not empty
            with st.spinner("Searching Google..."):
                try:
                    results = google_search_with_snippets(query)

                    if results:
                        st.subheader("Search Results:")
                        for result in results:
                            title = result.get('title', 'No title')
                            link = result.get('link', '#')
                            snippet = result.get('snippet', 'No description available.')

                            st.markdown(f"#### [{title}]({link})")
                            st.write(snippet)
                            st.caption(f"🔗 Source: {link}")
                            st.markdown("---")
                    else:
                        st.warning("No results found.")
                except Exception as e:
                    st.error(f"An error occurred while searching: {e}")
        else:
            st.warning("Please enter a valid search query.")


# --- System Utilities Menu ---
elif menu_choice == "System Utilities":
    st.header("System Utilities")
    util_option = st.selectbox("Choose a utility:", ["Read Ram", "To download a website's data"])

    if util_option == "Read Ram":
        st.subheader("💾 System Memory (RAM) Usage")
        mem_info = check_memory_usage()
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Memory", mem_info['Total'])
        col2.metric("Available Memory", mem_info['Available'])
        col3.metric("Used Memory", mem_info['Used'])
        col4.metric("Usage Percentage", mem_info['Percentage'])
        if st.button("Refresh"):
            st.experimental_rerun()

    if util_option == "To download a website's data":
        st.subheader("🕸️ Scrape Text from a Website")
        url = st.text_input("Enter a static website URL:")
        if st.button("Scrape Text"):
            if url:
                with st.spinner(f"Fetching content from {url}..."):
                    content = website_data(url)
                st.text_area("Scraped Content", content, height=400)
            else:
                st.warning("Please enter a URL.")
