# AC215 - Milestone4 - Fashion AI App

## Project Milestone 4 Organization


```
TODO: replace with file structure
```



## Overview

**Team Members**
Yushu Qiu, Weiyue Li, Daniel Nurieli, Michelle Tan

**Group Name**
The Fashion AI Group

**Project**
Our goal is to create an AI-powered platform that aggregates fashion items from various brands, allowing users to quickly and easily find matching items without the hassle of endless browsing. Using our App, the users can put in a request such as "find me a classic dress for attending a summer wedding" and receive the clothing item that matches their request most closely. 

## 1. Application Design Document

### Overview

We have designed this application to use a server to access our finetuned FashionClip model, a vector database, to take user text queries and return recommended fashion items with key metadata information. The application design and architecture choices reflect these user needs.

In the next milestone, we intend to deploy an LLM agent to improve the flexibility of how the users could interact with the Application.

---

### Solution Architecture

The Fashion AI application integrates key components to deliver a seamless user experience. During the application run, pinecone_service, vector_service, backend and frontend components are run. Users interact with the frontend application through a web browser, sending requests to the backend API for fashion recommendations and stylist interactions.

During the development cycle, we also have additional components that handle scraping, captioning, fintuning.

User interface will be covered in the frontend section below.

Here are the key components. each is a subfolder in the server folder:

1. **Frontend**: the frontend application is built using Next.js. It serves as the user interface for the Fashion AI application, allowing users to interact with the AI stylist and browse fashion items.

2. **Backend**: The backend directory contains the core API for the Fashion AI application. It handles requests from the frontend, processes user queries, and interacts with the database and external services to provide fashion recommendations.

3. **Pinecone service**: The pinecone_service directory contains the implementation for interacting with Pinecone, a vector database service used for managing and querying high-dimensional data. This service is essential for handling fashion item embeddings and similarity searches.

4. **Vector service**: The vector_service directory is responsible for managing vector representations of fashion items and performing similarity searches. This service leverages machine learning models to generate embeddings for fashion items, enabling efficient retrieval based on user queries.

---

### Technical Architecture

#### Technologies and Frameworks

1. **Frontend**:
   - **Next.js**: A React framework for building server-rendered applications, providing features like static site generation and API routes.
   - **React**: A JavaScript library for building user interfaces, enabling the creation of reusable components.
   - **Tailwind CSS**: A utility-first CSS framework for styling the application, allowing for rapid UI development.
   - **Axios**: A promise-based HTTP client for making API requests from the frontend.

2. **Backend**:
   - **Node.js**: A JavaScript runtime for building the backend API, allowing for asynchronous processing and scalability.
   - **Fast API**: A modern Python framework for creating high-performance APIs, used for handling AI model requests and processing user queries efficiently.

   - **Pytest**: A testing framework for writing and running unit tests to ensure the backend API's functionality and reliability.

3. **Deployment**:
   - **Docker**: A platform for developing, shipping, and running applications in containers, ensuring consistency across environments.

#### Design Patterns

- **MVC (Model-View-Controller)**: The application follows the MVC design pattern, separating concerns between the data model, user interface, and control logic.
- **Service Layer**: A service layer is implemented in the backend to handle business logic and API interactions, promoting code reusability and separation of concerns.
- **Component-Based Architecture**: The frontend is built using a component-based architecture, allowing for modular development and easier maintenance.

---

## 2. Frontend and API Documentation


### Application Components

#### Frontend Components

1. **Pages**: The application consists of various pages, each representing a different route:
   - **Home Page**: Introduces the application and provides navigation options.
   - **Stylist Page**: Users can interact with the AI fashion stylist for personalized recommendations.
   - **Gallery Page**: Showcases various fashion items and styles.

2. **Components**: Reusable UI components used across different pages:
   - **Buttons**: Custom button components for navigation and actions.
   - **Image Gallery**: Displays images in a grid format.
   - **Chat Input**: Component for user input in stylist interactions.

3. **Services**: Modules that handle API calls and data fetching, such as `DataService`, which communicates with the backend to retrieve fashion items and stylist recommendations.

4. **Styles**: The application uses Tailwind CSS for styling, allowing for responsive design and utility-first CSS classes.

#### API Components

The application utilizes a set of APIs to facilitate communication between the frontend and backend services. Key APIs include:

1. **Get Fashion Items**: Retrieves fashion items based on user queries.
2. **User Interaction with Stylist**: Allows users to send messages to the AI stylist and receive recommendations.

---

### Setup Instructions

To set up the Fashion AI application, follow these steps:

#### Frontend Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name/src/server/frontend
   ```

2. **Install Dependencies**:
   Ensure you have Node.js and npm installed. Then, install the required dependencies:
   ```bash
   npm install
   ```
<!-- 
3. **Environment Variables**:
   Create a `.env.production` file in the root of the frontend directory and add the necessary environment variables. Refer to the `.env.example` file for required variables.

4. **Build the Application**:
   Build the Next.js application for production:
   ```bash
   npm run build
   ``` -->

3. **Start the Development Server**:
   To start the application in development mode:
   ```bash
   npm run dev
   ```
   The application will be available at `http://localhost:3000`.

#### API Setup

1. **Backend Service**: Ensure that the backend service is running and accessible. The frontend relies on the backend APIs to function correctly.

2. **API Configuration**: Make sure the API base URL is correctly set in the environment variables. This is typically done in the `.env.production` file.

---

### Usage Guidelines

#### Frontend Usage

- **Navigating the Application**: Use the navigation buttons on the home page to access different sections.
- **Interacting with the Stylist**: On the Stylist page, describe your occasion and style preferences to receive personalized recommendations.
- **Viewing the Gallery**: The Gallery page allows browsing through various fashion items. Click on any item for more details.

#### API Usage

The function GetFashionItems posts the user query and gets items below in return to be serviced on the frontend. The url used is "/search".

    {
        "items": [
            {
                "item_name": "string",
                "item_caption": "string",
                "image_url": "string",
                "item_url": "string",
                "item_brand": "string",
                "item_type": "string",
                "rank": "number",
                "score": "number"
            }
        ]
    }

    
<!-- 
#### Error Handling

Implement error handling in the frontend to manage API response errors gracefully. Common error responses include:
- `400 Bad Request`: Invalid input data.
- `404 Not Found`: Requested resource does not exist.
- `500 Internal Server Error`: An unexpected error occurred on the server.

#### Testing

Use tools like Postman or cURL to test API endpoints during development. Ensure that all endpoints are functioning as expected before deploying changes.
 -->


---
---
---

****Requirements****


 Description of application components, setup instructions, and usage guidelines.


1. **Application Design Document**:
   - A detailed document outlining the application’s architecture, user interface, and code organization.
   - **Should Include**:
     - **Solution Architecture**: High-level overview of system components and their interactions.
     - **Technical Architecture**: Specific technologies, frameworks, and design patterns used.

2. **APIs & Frontend Implementation**:
   - Working code for APIs and the front-end interface.
   - **Should Include**:
     - **GitHub Repository**: All source code with logical organization and proper documentation.
     - **README File**: Description of application components, setup instructions, and usage guidelines.

3. **Continuous Integration Setup**:
   - A functioning CI pipeline that runs on every push or merge.
   - **Pipeline Must Include**:
     - **Code Build and Linting**: Automated build process and code quality checks using linting tools (e.g., ESLint, Flake8) running on GitHub Actions.
     - **Automated Testing**: Execution of unit, integration, and system tests with test results reported.

4. **Automated Testing Implementation**:
   - Integration of automated tests within the CI pipeline using GitHub Actions.
   - **Should Include**:
     - **Unit Tests**: For individual components and functions.
     - **Integration Tests**: For integrating multiple components.
     - **System Tests**: Covering user flows and interactions.
     - **Test Coverage Reports**: Integrated into the CI pipeline to monitor code coverage to be at least 50%.

5. **Test Documentation**:
   - Detailed explanations of the testing strategy and implemented tests.
   - **Should Include**:
     - **Testing Tools Used**: (e.g., PyTest)
     - **Instructions to Run Tests Manually**: For developers to replicate test results locally.
