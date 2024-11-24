function sendMessage() {
    var userMessage = document.getElementById("user-input").value;
    if (userMessage.trim() === "") return;

    appendMessage("user", userMessage);
    document.getElementById("user-input").value = "";

    // Call Rasa API to get chatbot response
    fetchRasaResponse(userMessage);
}

function appendMessage(sender, message) {
    var chatDisplay = document.getElementById("chat-display");
    var messageElement = document.createElement("div");
    messageElement.className = sender;
    messageElement.innerHTML = `<strong>${sender}:</strong> ${message}`;
    chatDisplay.appendChild(messageElement);
    chatDisplay.scrollTop = chatDisplay.scrollHeight;
}

function fetchRasaResponse(userMessage) {
    // Use your Rasa server endpoint
    var rasaEndpoint = "http://ec2-18-144-156-116.us-west-1.compute.amazonaws.com:5005/socket";

    fetch(rasaEndpoint, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({
            sender: "user",
            message: userMessage,
        }),
    })
    .then(response => response.json())
    .then(data => {
        // Display chatbot response
        if (data && data.length > 0) {
            appendMessage("chatbot", data[0].text);
        }
    })
    .catch(error => console.error("Error fetching Rasa response:", error));
}
