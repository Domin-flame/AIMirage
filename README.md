# 🎬 AI Marketing Avatar

An AI-powered platform that transforms photos and text into engaging marketing videos using state-of-the-art open-source generative AI models.

> **Status:** 🚧 Work in Progress

## 📖 Overview

AI Marketing Avatar is an experimental project aimed at simplifying the creation of marketing videos through artificial intelligence.

Instead of requiring expensive cameras, actors, or video editing software, the platform will allow users to:

* Upload a portrait photo
* Enter a marketing script
* Generate an AI-powered talking avatar
* Export a ready-to-share video

The long-term vision is to provide businesses, entrepreneurs, and content creators with an accessible tool for producing high-quality promotional videos.

---

## ✨ Features

### Current Features

* ✅ FastAPI backend
* ✅ Interactive web interface
* ✅ AI image generation from text prompts
* ✅ Optimized inference pipeline
* ✅ Image preview and download

### Planned Features

* 🚧 Image upload
* 🚧 AI avatar generation
* 🚧 Text-to-speech integration
* 🚧 AI video generation
* 🚧 Video download
* 🚧 User authentication
* 🚧 Generation history
* 🚧 Cloud deployment

---

## 🛠️ Tech Stack

### Backend

* Python
* FastAPI
* PyTorch
* Diffusers
* Stable Diffusion Turbo

### Frontend

* HTML
* CSS
* JavaScript

### AI Models

Current:

* Stable Diffusion Turbo

Planned:

* SadTalker
* MuseTalk (future research)
* Text-to-Speech engine

---

## 🏗️ Project Architecture

```text
Frontend
     │
     ▼
 FastAPI Backend
     │
     ▼
 Video Engine
     │
     ├── Stable Diffusion (Images)
     ├── SadTalker (Talking Avatar)
     └── Future AI Models
```

---

## 🎯 Project Goal

This project focuses on building a practical AI-powered platform rather than training large-scale AI models from scratch.

The objective is to integrate powerful open-source AI technologies into a user-friendly application capable of generating marketing-ready videos.

---

## 🚀 Roadmap

### Phase 1

* [x] Build FastAPI backend
* [x] Integrate Stable Diffusion
* [x] Optimize inference speed

### Phase 2

* [ ] Upload portrait images
* [ ] Install and integrate SadTalker
* [ ] Generate talking avatar videos

### Phase 3

* [ ] Text-to-Speech integration
* [ ] Multiple voice selection
* [ ] Improved UI/UX

### Phase 4

* [ ] User authentication
* [ ] Cloud deployment
* [ ] API version
* [ ] Commercial MVP

---

## 💻 Installation

> **Requirements**

* Python 3.10
* Git
* FFmpeg
* Virtual Environment

Clone the repository:

```bash
git clone https://github.com/your-username/AI-Marketing-Avatar.git
cd AI-Marketing-Avatar
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it:

### Windows

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the backend:

```bash
uvicorn main:app --reload
```

---

## 📸 Screenshots

Coming soon...

---

## 📚 Motivation

This project started as a technical challenge to explore the potential of generative AI in content creation.

Beyond experimenting with AI models, the goal is to understand how software engineering, backend development, and artificial intelligence can be combined to create a real-world product.

---

## 🤝 Contributing

Contributions, suggestions, and discussions are welcome.

Feel free to open an issue or submit a pull request.

---

## 📄 License

This project is currently under development.

A license will be added before the first stable release.
