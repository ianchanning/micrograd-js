import { defineConfig } from "eslint/config";
import prettierPlugin from "eslint-plugin-prettier"; // Renamed to avoid conflict with config
import prettierConfig from "eslint-config-prettier"; // Import the config that disables conflicting rules
import globals from "globals";
import path from "node:path";
import { fileURLToPath } from "node:url";
import js from "@eslint/js";
import { FlatCompat } from "@eslint/eslintrc";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const compat = new FlatCompat({
  baseDirectory: __dirname,
  recommendedConfig: js.configs.recommended,
  allConfig: js.configs.all,
});

export default defineConfig([
  {
    extends: compat.extends("eslint:recommended"),

    plugins: {
      prettier: prettierPlugin, // Use the renamed plugin import
    },

    languageOptions: {
      globals: {
        ...globals.browser,
        ...globals.jest, // For Jest test files, if not handled by a separate config
      },

      ecmaVersion: 2023,
      sourceType: "module",
    },

    rules: {
      // These rules are now handled by Prettier via 'prettier/prettier'
      // and disabled by 'eslint-config-prettier'.
      // Keeping them here would cause conflicts.
      // indent: ["error", 2],
      // "linebreak-style": ["error", "unix"],
      "prettier/prettier": "error", // This rule runs Prettier as an ESLint rule
      // quotes: ["error", "double"],
    },
  },
  // Add eslint-config-prettier at the end to ensure it overrides
  // any conflicting formatting rules from previous configurations.
  prettierConfig,
]);
