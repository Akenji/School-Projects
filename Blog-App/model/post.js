const mongoose=require('mongoose')
const postSchema=new mongoose.Schema({
    title:{type:String,required:[true,"Post title must be provided"]},
    body:{type:String,required:[true,"Post body must be provided"]},
    image:{type:String,required:[true,"image url must be provided"]},
    postedBy:{type:String,required:[true,"This field can't be empty"]}
},{timestamps:true})

module.exports=mongoose.model('post',postSchema)