document.getElementById("search-button").addEventListener("click", async () => {
    const queryText = document.getElementById("search-input").value;
    const resultsContainer = document.getElementById("results");

    if (!queryText) {
        alert("Please enter a search query.");
        return;
    }

    resultsContainer.innerHTML = "<p>Loading...</p>";

    try {
        const response = await fetch("http://localhost:8000/search", {  // Updated to point to backend
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ queryText }),
        });

        const results = await response.json();
        resultsContainer.innerHTML = "";

        if (results.length === 0) {
            resultsContainer.innerHTML = "<p>No results found.</p>";
        } else {
            results.sort((a, b) => b.score - a.score);
            results.forEach((result, index) => {
                const resultItem = document.createElement("div");
                resultItem.className = "result-item";
                resultItem.innerHTML = `
                    <img src="${result.metadata.url}" alt="Fashion item ${result.id}">
                    <div>
                        <h3>Rank ${index + 1}</h3>
                        <p>Image ID: ${result.id}</p>
                        <p>Score: ${result.score.toFixed(2)}</p>
                    </div>
                `;
                resultsContainer.appendChild(resultItem);
            });
        }
    } catch (error) {
        console.error("Error fetching results:", error);
        resultsContainer.innerHTML = "<p>Error fetching results.</p>";
    }
});
