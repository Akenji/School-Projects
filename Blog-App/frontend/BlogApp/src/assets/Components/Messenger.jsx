import { useContext, useEffect, useState } from 'react'
import {io} from 'socket.io-client'
import { AuthContext } from '../Context/AuthContext'
import MessageUser from './MessageUser'
const Messenger = () => {
    const {userState,users}=useContext(AuthContext)
    // const [mess,setMess]=useState("")
    // const [message,setMessage]=useState("")
    // const [messageR,setMessageR]=useState("")
    // const [show,setShow]=useState(false)
    // const socket=io('http://localhost:5000')
    // document.windowId=Math.round(Math.random()*10000)
    // const sendMessage=(e)=>{
    //     e.preventDefault()
    //     let newElement=document.createElement('div')
    //     newElement.classList.add("col-10","d-flex","justify-content-end")
    //     newElement.innerHTML=`
    //         <p class='bg-custom text-display-self d-flex justify-content-end p-3 my-2'>${message}</p>
    //     `
    //     let textField=document.getElementById('text-field')
    //     let display=document.getElementById('display').insertBefore(newElement,textField)
    //     socket.emit("send_message",{message:message,windowId:document.windowId})
    // }
    // socket.once("receive_message",(data)=>{
    //         if(data.windowId!=document.windowId) {
    //             setMessageR(data.message)
    //             let newElement=document.createElement('div')
    //             newElement.classList.add("col-12","d-flex","justify-content-start")
    //             newElement.innerHTML=`
    //                 <p class='bg-custom-2 text-display d-flex justify-content-start p-3 my-2'>${data.message}</p>
    //             `
    //             let textField=document.getElementById('text-field')
    //             let display=document.getElementById('display').insertBefore(newElement,textField)
    //         }
    //     })
    return ( 
        <div className="my-5">
        {
            users &&
            users.map((user,e)=>{
             return <MessageUser user={user} key={e} />
        })}
        </div>
     );
}
 
export default Messenger;