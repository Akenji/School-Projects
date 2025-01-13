import { useState } from 'react'
import './App.css'
import Navbar from './assets/Components/Navbar'
import Home from './assets/Components/Home'
import  {BrowserRouter as  Router,Routes,Route}  from 'react-router-dom'
import Form from './assets/Components/Form'
import AuthContextProvider from './assets/Context/AuthContext'
import Messenger from './assets/Components/Messenger'
import Details from './assets/Components/Details'


function App() {
  return (
      <AuthContextProvider>
        <Router>
          <Navbar />
          <Routes>
            <Route path='/' element={<Home />} />
            <Route path='/messenger' element={<Messenger />} />
            <Route path='/register' element={<Form mode="register" />} />
            <Route path='/login' element={<Form mode="login" />} />
            <Route path='/details/:id' element={<Details />} />
          </Routes>
        </Router>
      </AuthContextProvider>
  )
}

export default App
