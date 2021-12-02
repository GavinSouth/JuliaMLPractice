#' Notes:
#' Julia: Execute Code (Ctrl+Enter)
#' Julia: Execute Code Block (Alt+Enter)
#' Julia: Execute Code Cell (Shift+Enter)

#import Pkg; Pkg.add("CSV")
#import Pkg; Pkg.add("DataFrames")
#import Pkg; Pkg.add("Plots")
#import Pkg; Pkg.add("BenchmarkTools")
#import Pkg; Pkg.add("StatsBase")
#import Pkg; Pkg.add("Pipe")
#import Pkg; Pkg.add("Flux")

#using Meshing
#using GeometryTypes
#using LinearAlgebra: dot, norm
#using FileIO


enerate_real_data(train_size)
fake = generate_fake_data(train_size)

scatter(real[1,1:500],real[2,1:500])
scatter!(fake[1,1:500],fake[2,1:500]) # adding the explanation point will let you layer multiple plots


#' Model building
function NeuralNetwork()
    return Chain(
            Dense(2, 25,relu),
            Dense(25,1,x->σ.(x))
            )
end

#' training a CNN
#' Concatenate along dimension 2 (cbind)
#' Or in other words, it's making a data frame between the fake and not fake.
X    = hcat(real,fake) 

#' Concatenate along dimension 1 (rbind)
#' Making an array of ones for true values, and then the same for fakes.
Y    = vcat(ones(train_size),zeros(train_size))

#' Organize our data into one single dataset. 
#' We use the DataLoader function from Flux, that helps us create the batches and shuffles our data. 
data = Flux.Data.DataLoader((X, Y'), batchsize=100,shuffle=true);

#' Defining our model, optimization algorithm and loss function
#' Then, we call our model and define the loss function and the optimization algorithm. 
#' In this example, we are using gradient descent for optimization and cross-entropy for the loss function.
m   = NeuralNetwork()
opt = Descent(0.05)
loss(x, y) = sum(Flux.Losses.binarycrossentropy(m(x), y))

#' Training model
ps = Flux.params(m)
epochs = 20
for i in 1:epochs # looping through a range
    Flux.train!(loss, ps, data, opt)
end
println(mean(m(real)),mean(m(fake))) # Print model prediction


#' Visualizing predictions
scatter(real[1,1:100],real[2,1:100],zcolor=m(real)')
scatter!(fake[1,1:100],fake[2,1:100],zcolor=m(fake)',legend=false)


#' DATA FRAME MANIPULATION

# Changing the data shape that is printed out
ENV["LINES"] = 20
ENV["COLUMNS"] = 20

dd = CSV.read("TSLA.csv", DataFrame)

# Looking through the data
size(dd)
describe(dd) # This gives us an idea of the data. 

plot(dd.Volume)
p1 = plot(dd.Date, dd.Open)
p2 = plot(dd.Date, dd.Volume)

plot(p1, p2, layout = (2,1))