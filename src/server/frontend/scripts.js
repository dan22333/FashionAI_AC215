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
            description.textContent = results.description;
            resultsContainer.appendChild(description);

            // Add the items
            results.items.forEach((item) => {
                const resultItem = document.createElement("div");
                resultItem.className = "result-item";
                resultItem.innerHTML = `
                    <img src="${item.item_url}" alt="${item.item_name}" />
                    <div>
                        <h3>${item.item_name}</h3>
                        <p>Brand: ${item.item_brand}</p>
                        <p>${item.item_caption}</p>
                        <p>Score: ${item.score}</p>
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
