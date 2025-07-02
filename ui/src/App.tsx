import TrainerContainer from './components/trainer_container/TrainerContainer'
import './App.css'
import Nav from './components/Nav'
import React from 'react'

function App() {

  return (
    <>
      <div className='h-auto w-[100vw] bg-gray-900'>
        <Nav />  
        <TrainerContainer/>
      </div>
    </>
  )
}

export default App
