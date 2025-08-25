// pinecone.js
require("dotenv").config();
const axios = require("axios");
const { pipeline } = require("@xenova/transformers");

console.log("Starting Pinecone query system with local embeddings...");

// Configuration
const PINECONE_API_KEY = process.env.PINECONE_API_KEY;
const PINECONE_INDEX_URL = process.env.PINECONE_INDEX_URL;

// Validate required environment variables
if (!PINECONE_INDEX_URL) {
  console.error("❌ ERROR: PINECONE_INDEX_URL is not set in .env file");
  console.error("Check your Pinecone dashboard for the correct URL");
  process.exit(1);
}

if (!PINECONE_API_KEY) {
  console.error("❌ ERROR: PINECONE_API_KEY is not set in .env file");
  console.error("Check your Pinecone dashboard for the correct API key");
  process.exit(1);
}

// Configure axios with timeout for large model downloads
axios.defaults.timeout = 300000; // 5 minutes for model download

// Initialize the embedding model (this is async)
let embedder = null;

async function initEmbedder() {
  if (embedder) return embedder;

  console.log(
    "🔤 Loading embedding model (this may take a moment on first run)..."
  );
  try {
    embedder = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2");
    console.log("✅ Embedding model loaded successfully");
    return embedder;
  } catch (error) {
    console.error("❌ Failed to load embedding model:", error.message);

    if (error.message.includes("Failed to fetch")) {
      console.error("\n⚠️ Network issue: Check your internet connection");
      console.error(
        "The model needs to be downloaded (about 80MB) on first run"
      );
    } else if (error.message.includes("not found")) {
      console.error(
        "\n⚠️ Model not found: Check if 'Xenova/all-MiniLM-L6-v2' is available"
      );
    } else if (error.code === 'ECONNABORTED') {
      console.error("\n⚠️ Timeout: Model download took too long. Try again with a better connection.");
    }

    throw error;
  }
}

// Generate 384D embedding using local model
async function getEmbedding(text) {
  try {
    const embedder = await initEmbedder();

    console.log(
      `🔤 Generating embedding for: "${text.substring(0, 30)}${
        text.length > 30 ? "..." : ""
      }"`
    );

    const output = await embedder(text, {
      pooling: "mean",
      normalize: true,
    });

    // CRITICAL FIX: output.data is ALREADY the embedding vector (384 dimensions)
    // We don't need [0] indexing - that was causing the 0D error
    const embedding = Array.from(output.data);

    console.log(
      `✅ Generated ${embedding.length}D embedding (matches your index)`
    );
    return embedding;
  } catch (error) {
    console.error("❌ Failed to generate embedding:", error.message);
    throw error;
  }
}

// Query Pinecone with proper parameter names and optional namespace
async function queryPinecone(queryText, topK = 3) {
  try {
    const vector = await getEmbedding(queryText);

    console.log(`📡 Sending query directly to: ${PINECONE_INDEX_URL}/query`);

    // Optionally set namespace via env (only include if set)
    const namespace = process.env.PINECONE_NAMESPACE || undefined;
    const body = {
      vector: vector,
      topK: topK,
      includeMetadata: true,   // CORRECT param name (not includeMeta)
      includeValues: false,    // usually you don't need the dense vectors back
    };
    
    if (namespace) {
      body.namespace = namespace;
      console.log(`🏷️ Using namespace: ${namespace}`);
    }

    const response = await axios.post(
      `${PINECONE_INDEX_URL}/query`,
      body,
      {
        headers: {
          "Api-Key": PINECONE_API_KEY,
          "Content-Type": "application/json",
        },
        timeout: 30000, // 30 second timeout for queries
      }
    );

    return response.data;
  } catch (error) {
    console.error("❌ Query failed:", error.message);
    if (error.response) {
      console.error(`Status: ${error.response.status}`);
      console.error(`Response: ${JSON.stringify(error.response.data, null, 2)}`);
      
      // Specific error handling
      if (error.response.status === 401) {
        console.error("🔑 Authentication failed - check your API key");
      } else if (error.response.status === 404) {
        console.error("🔍 Index not found - check your index URL");
      } else if (error.response.status === 400) {
        console.error("📊 Bad request - check vector dimensions match your index");
      }
    } else if (error.code === 'ECONNABORTED') {
      console.error("⏰ Query timeout - Pinecone might be slow to respond");
    }
    throw error;
  }
}

// Main execution function with proper error handling
async function main() {
  try {
    // Initialize embedder first to catch loading errors early
    console.log("🚀 Initializing embedding model...");
    await initEmbedder();
    
    const query = "developed by open ai ";
    console.log(`\n🔍 Querying Pinecone for: "${query}"`);

    const results = await queryPinecone(query);

    console.log("\n🎯 Top matches:");
    if (!results.matches || results.matches.length === 0) {
      console.log("No results found. Check if your index has data.");
      console.log("💡 Tip: Make sure you've uploaded vectors to your Pinecone index");
      return;
    }

    results.matches.forEach((match, i) => {
      console.log(`${i + 1}. Score: ${match.score.toFixed(4)}`);
      console.log(`   ID: ${match.id}`);

      // Defensive metadata printing (handles different shapes)
      if (match.metadata) {
        // Common cases: metadata.content, metadata.text, or any other field
        const text = match.metadata.content || 
                    match.metadata.text || 
                    match.metadata.chunk ||
                    JSON.stringify(match.metadata);
        console.log(`   Text: ${text}`);
      } else if (match.meta) {
        // Alternative metadata field name
        const text = match.meta.content || 
                    match.meta.text ||
                    JSON.stringify(match.meta);
        console.log(`   Text: ${text}`);
      } else {
        console.log("   Text: ⚠️ No metadata returned for this match");
      }

      console.log(); // blank line for readability
    });

    // Optional: Show total matches found
    console.log(`📊 Found ${results.matches.length} matches`);

    // Uncomment below line if you need to inspect the raw response structure
    // console.log("🔍 RAW RESPONSE:", JSON.stringify(results, null, 2));

  } catch (error) {
    console.error("\n💥 Critical error in main():", error.message);
    
    // Provide helpful troubleshooting tips based on error type
    if (error.message.includes("Failed to load embedding model")) {
      console.error("\n🔧 Troubleshooting tips:");
      console.error("1. Check your internet connection");
      console.error("2. Try running again (model downloads on first use)");
      console.error("3. Make sure you have enough disk space (~200MB)");
    } else if (error.message.includes("Query failed")) {
      console.error("\n🔧 Troubleshooting tips:");
      console.error("1. Verify your PINECONE_API_KEY and PINECONE_INDEX_URL in .env");
      console.error("2. Check if your index exists in Pinecone dashboard");
      console.error("3. Ensure your index has the correct dimensions (384 for this model)");
    }
    
    process.exit(1);
  }
}

// Graceful shutdown handling
process.on('SIGINT', () => {
  console.log('\n👋 Shutting down gracefully...');
  process.exit(0);
});

process.on('unhandledRejection', (reason, promise) => {
  console.error('💥 Unhandled Rejection at:', promise, 'reason:', reason);
  process.exit(1);
});

// Execute the main function
console.log("🔄 Starting application...");
main().catch((error) => {
  console.error("💥 Unhandled error:", error);
  process.exit(1);
});