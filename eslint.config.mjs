import { defineConfig } from "eslint/config";
import prettier from "eslint-plugin-prettier";
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
      prettier,
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
      indent: ["error", 2],
      "linebreak-style": ["error", "unix"],
      "prettier/prettier": "error",
      quotes: ["error", "double"],
    },
  },
]);

