document.getElementById("search-button").addEventListener("click", async () => {
    const queryText = document.getElementById("search-input").value;
    const resultsContainer = document.getElementById("results");

    if (!queryText) {
        alert("Please enter a search query.");
        return;
    }

    resultsContainer.innerHTML = "<p>Loading...</p>";

    try {
        const response = await fetch("http://localhost:8000/search", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ queryText }), // Send queryText as JSON
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const results = await response.json();
        resultsContainer.innerHTML = ""; // Clear the results container

        // Check if results.items exists and is an array
        if (!Array.isArray(results.items) || results.items.length === 0) {
            resultsContainer.innerHTML = "<p>No results found.</p>";
        } else {
            // Sort the items by score
            results.items.sort((a, b) => b.score - a.score);

            // Add the description
            const description = document.createElement("p");
            description.textContent = results.description || "No description available";
            resultsContainer.appendChild(description);

            // Add the items
            results.items.forEach((item) => {
                const resultItem = document.createElement("div");
                resultItem.className = "result-item";
                resultItem.style.border = "1px solid #ddd";
                resultItem.style.padding = "10px";
                resultItem.style.marginBottom = "10px";
                resultItem.style.borderRadius = "4px";

                resultItem.innerHTML = `
                    <div style="display: flex; align-items: flex-start; gap: 10px;">
                        <img src="${item.image_url || "default.jpg"}" alt="${item.item_name || "Unknown Name"}" style="max-width: 100px; border-radius: 4px;" />
                        <div>
                            <h3>${item.item_name || "Unknown Name"}</h3>
                            <p><strong>Brand:</strong> ${item.item_brand || "Unknown Brand"}</p>
                            <p><strong>Gender:</strong> ${item.item_gender || "Unknown Gender"}</p>
                            <p><strong>Type:</strong> ${item.item_type || "Unknown Type"}</p>
                            <p><strong>Sub-Type:</strong> ${item.item_sub_type || "Unknown Sub-Type"}</p>
                            <p><strong>Caption:</strong> ${item.item_caption || "No caption available"}</p>
                            <p><strong>Rank:</strong> ${item.rank || "N/A"}</p>
                            <p><strong>Score:</strong> ${item.score || "N/A"}</p>
                            <a href="${item.item_url || "#"}" target="_blank" style="color: #007bff; text-decoration: none;">View Item</a>
                        </div>
                    </div>
                `;
                resultsContainer.appendChild(resultItem);
            });
        }
    } catch (error) {
        console.error("Error fetching results:", error);
        resultsContainer.innerHTML = "<p>Error fetching results. Please try again later.</p>";
    }
});
