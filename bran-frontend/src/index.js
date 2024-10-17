import React from "react";
import ReactDOM from "react-dom";
import App from "./App";
import { createTheme, ThemeProvider } from "@mui/material/styles";
import CssBaseline from "@mui/material/CssBaseline"; // Importar CssBaseline para aplicar o tema escuro

const theme = createTheme({
  palette: {
    mode: "dark", // Define o tema como escuro
    primary: {
      main: "#90caf9", // Azul claro para o tema escuro
    },
    secondary: {
      main: "#f48fb1", // Rosa claro para contraste
    },
    background: {
      default: "#121212", // Fundo escuro padrão
      paper: "#1e1e1e", // Fundo para papéis
    },
    text: {
      primary: "#ffffff", // Texto principal em branco
    },
  },
  typography: {
    fontFamily: "Roboto, sans-serif",
    h4: {
      fontWeight: 600, // Um toque de peso extra para os títulos
    },
    h6: {
      fontWeight: 500,
    },
  },
});

ReactDOM.render(
  <ThemeProvider theme={theme}>
    <CssBaseline /> {/* Aplica estilos globais para o tema escuro */}
    <App />
  </ThemeProvider>,
  document.getElementById("root"),
);
