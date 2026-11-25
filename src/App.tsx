import { useState } from 'react'
import './App.css'

function App() {
  const [titulo, setTitulo] = useState('')
  const [contenido, setContenido] = useState('')

  return (
    <main className="editor">
      <header>
        <h1>Edición de contenido</h1>
        <p className="marca">joe.network</p>
      </header>

      <section className="panel">
        <label>
          Título
          <input
            value={titulo}
            onChange={(e) => setTitulo(e.target.value)}
            placeholder="Escribe un título…"
          />
        </label>
        <label>
          Contenido
          <textarea
            value={contenido}
            onChange={(e) => setContenido(e.target.value)}
            placeholder="Escribe tu contenido…"
            rows={10}
          />
        </label>
      </section>

      <section className="panel vista-previa">
        <h2>Vista previa</h2>
        <h3>{titulo || 'Sin título'}</h3>
        <p>{contenido || 'Aún no hay contenido.'}</p>
      </section>
    </main>
  )
}

export default App
