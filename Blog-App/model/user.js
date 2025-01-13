const mongoose=require('mongoose')
const bcrypt=require('bcryptjs')
const userSchema=new mongoose.Schema({
    googleId:String,
    username:{type:String,required:[true,"Username must be provided"]},
    email:{type:String,required:[true,"Email must be provided"],unique:[true,"User with that email already exist!"]},
    password:{type:String,required:[true,"Password must be provided"]},
    tasks:{type:[{
        title:String,
        body:String,
        image:String,
    }]}
})
userSchema.pre('save',async function (next){
    const salt=bcrypt.genSaltSync(10)
    const newPassword=bcrypt.hashSync(this.password,salt)
    this.password=newPassword
    next()
})
userSchema.statics.login=async function (email,password){
    let oldUser=await this.findOne({email})
    if(!oldUser){
        return null
        //throw Error("No such user. Create account")
    }
    if(bcrypt.compareSync(password,oldUser.password)){
        return oldUser
    }else{
        throw Error("Wrong Password")
    }
}
module.exports=mongoose.model('users',userSchema)