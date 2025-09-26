I scrapped a bit Youtube, because we believe that our GyroSI is superior to all other architeectures, but we need to remain grounded on what other models trully offer and find out how our theory provides a better route - , so in implementation we know which elements need to be our main architecture, which would be external helpers, etc.  We need to consider if we are missing something, if we really are not missing anything, etc.
===
YOUTUBE AI related Scripts:

Part 1:
 we ned to properly understand how, , map our functions to theirs, and consider if there are gaps or a
Hello, how's it going? Today we're going to look at this paper right here. Energy- based transformers are scalable
learners and thinkers. This paper combines the uh general concept of
energybased models with transformers and suggests a learning paradigm for it and
shows how that learning paradigm applies and how it can be made scalable. uh with
it they do achieve some pretty promising results in that the experiments are on
smaller scale data but in terms of language modeling uh video modeling and
things like this the scaling trends show that this actually might be a more
scalable architecture and a more promising architecture once you do go large scale. So I would say this is a
pretty interesting direction of research and uh promising although not uh
obviously verified yet at scale but I think that's what does makes it make it
interesting. It's a bit different from what you're used to but it also goes back to concepts that have been around
for a while and I'm very happy to see more energy based modeling work here. So
let's dive in. Uh the general concept that the authors here ask are is it
possible to generalize system to thinking approaches and develop models
that learn to think solely from unsupervised learning. Very big and bold
question. Obviously uh we're not talking about can we sort of build better models
or something like this like no we are going to um
system two thinking which is sort of the more logical slow thinking that that
humans do and we're going to do that in machines and they here largely refer to
inference time computation techniques and we'll get to that in a bit. So they're asking
can we teach computers how to do system two thinking and can we do so without
any supervision so from purely unsupervised learning and in a way that
is generalizing again very very big question. So by system two thinking
techniques obviously if you um know from cognitive science uh system one thinking
broadly refers to sort of quick um quick and intuitive thinking.
Now this is what how you do most of your day. uh if you how you go about it. If
you're sort of I don't know want want to eat a banana like almost all of your
actions are going to be system one thinking. You sort of subconsciously walk to where you store fruit. You
subconsciously grab one. You don't have to think about walking about extending your muscles. You may not even have to
think of of how to, you know, peel and eat and so on. All of this is just super subconscious. it's ingrained and you
don't have to think about it. However, system two thinking kicks in whenever
system one thinking is at its limits. Although, I'm going to guess it's debatable exactly what the difference is
and if system whether system two thinking is sort of an extension of system one thinking, whether there's a
spectrum or whether this is something completely different that you know is is
is is totally separate and the two things work together. I have no idea.
But in general, system two thinking is characterized by being slow and being explicit.
So whenever you sort of in your mind talk to yourself and then sort of work
through a logical series of steps to arrive at a conclusion, that would be
system 2 thinking. And it reasonably said uh the authors here
say most of machine learning so far has been largely in the domain of system one thinking. There is a bit of debate about
that. So some people show that for example if you have a transformer and
that transformer has many many layers and you put a piece of language down
here and up spits the next token prediction. In here you have multiple
steps of computation given by layer by layer by layer and therefore it's it's
completely conceivable that there is one or another form of quote unquote thinking happening there however you
want to define that. However, um largely you could say that sort of building a
model and then doing a forward pass and taking its output that's kind of analogous to the system system one
thinking whereas system two thinking is much more um if you sort of do that
explicit. So let's say you do have that transformer but you use it to do chain
of thought prompting. So you input something and the transformer outputs thinking tokens that refer back to
itself obviously. So not refer back to itself you auto reggressively feed the
thinking tokens into the transformer and and so on. So you would produce an entire sequence. You use the transformer
over and over and over again and you have some explicit thinking going on
right here and eventually your output would uh rely
on the fact that you have done that thinking. The authors here largely go for the
approach that where they say the moment you you use a trained model at inference
time more than once sort of the moment you put more computation into um getting
an answer from the model than simply doing a single forward pass. That is
part of what they call thinking. Again, thinking is a big word, but to them, and
that's only really at the end of the paper, thinking is actually not even an act. It's a metric. And their metric of
thinking is really sort of how much better can we get by doing multiple forward passes through models rather
than a single forward pass. You might already know a couple of models that do this apart from autogressive
transformers. So for example, recurrent neural networks
sort of intuitively do some some sort of multiple forward passes, although you could debate that
and say processing an entire sequence is simply one forward pass. Other models
you could say are diffusion. So diffusion models uh also do multiple
forward passes through a model in order to arrive at their output. Again, you
might debate that and say, "No, no, no. Actually, that's just that's just one forward one overarching forward pass.
That's just kind of one inference through the model is well, you do all these computations." Um, so again, all
of this is debatable, but I'll leave it at that. You make your own decisions
about what thinking is and what it means and which models are doing thinking and which ones aren't. This paper right here
is looking at energy based models. And the way you do inference in energy
based models is that you do multiple forward passes. And that's why they call
their energy- based transformer variants thinkers because it does that because
you can put more compute in and you get a quote unquote better answer.
All right. Yeah. Here this is a bit illustrated. So
in an autogressive transformer you would simply have your sequence uh and you would predict the next token. In an RNN
you do do multiple computations but again you take your sequence you predict the next token here. A diffusion
transformer um obviously does some reverse inference here. and an energy
based transformer. We'll get to that in one second.
They also talk about sort of the current trend of reasoning models and those
reasoning models again they largely are trained with uh chain of thought prompting and things like that you using
reinforcement learning. So they're saying that um this usually only works
in domains where rule-based rewards can easily verify answers such as math and coding. Uh so you are only it's only
applicable to a small range of problem types meaning that um yes you can do
reinforcement learning but you somehow have to give a reward and that reward usually is derived algorithmically. So
you give it a math puzzle and um you already know the answer or you give it a
coding um puzzle and you can verify that the unit tests pass or that it compiles
or that it gives a correct output. So reinforcement learning is fantastic. But other than that, if you don't have
supervised data, you're kind of out of luck. So we need things um we need to
train these models in a way that they are able to do inference time compute um
without necessarily relying on reinforcement learning. Although what you'll see is that um even though they
contrast what they're doing with all of these models right here, there's actually not nothing stopping us from
combining the two. To me, reasoning as the models currently do, chain of
thought and all of that is completely orthogonal of what these energy based
models do. And it's quite conceivable that the energy- based models are also
being used to do chain of thought. All right. Again, their central question, can we rely entirely on
unsupervised learning to develop system 2 thinking? This uh such a capability would enable
generalization of current system to thinking approaches to any problem, any modality and avoid the reliance on
external human reward or model supervision. for that they say okay there are three
key facets of system two thinking and from now on when you hear system two thinking just sort of think the goals of
there like I'm what I'm pretty sure and I obviously have no proof of that what I'm pretty sure is they started from the
models they first developed energy based transformers because energy- based models are an idea and transformers are
an idea and you combine them you define energy- based transformers and then they sort of worked backwards and said what
are the properties of these models and oh those just magically happen to be uh
the three key facets of system 2 thinking right I don't really buy that
uh I think there was some reverse engineering happening right here so just take these as three nice properties of
energy- based models rather than system two thinking dynamic allocation of
computation in an energy- based model and you'll see that you can choose how
much computation you want to invest into inference time. Um so if you have an
energy based transformer and that is um a model on language and you wanted to
predict the next token, you are not limited to just one forward pass. You
can put in more computation and get a more accurate distribution over what
that next token should be. Facet two, modeling uncertainty in continuous state
spaces. Again, this is a property of energy- based models that they can give
you a a notion of uncertainty uh in their in their predictions. And so, EBMs
naturally can naturally model uncertainty without having to model exact likelihoods by modeling the
relative unnormalized likelihoods of predictions.
As the real world often contains many inherently unpredictable elements, for instance, when a pedestrian might emerge
from behind a parked vehicle, the ability to express uncertainty in predictions is essential to being cautious and is a natural capability of
humans. And then facet three, verification of predictions. So in an energy based model
we are doing a little bit uh what you might be used to from from GANs from
generative adversarial models where there is a verifier or a discriminator involved. So a model that can assess how
good quote unquote something is. And so naturally energy- based models have this
inherent ability to not just predict but judge uh predictions.
So how do we do this? Um here again you will have uh an an inference across an
energy based models. You can see this is next token prediction. So you do have your context. The dog caught the and
then the question is in an energy- based model, how do you how do you get a prediction out for the next token? And
the way we're going to do this is we're not just going to predict a token, but we're going to output an entire
distribution over tokens. Now, this is still the same as in a classic transformer that it also outputs a
distribution over tokens, but the way we do it is different. The way we do it is we're actually going to start with a
completely random distribution and then we're going to step by step refine that
distribution and and sort of shape that distribution until we end up with our
final distribution. And again, we can choose to do this for longer and get a more accurate um output or we can choose
to do this for shorter and get a less accurate output. This again um might
remind you of of diffusion models but it's a bit different of how we do it here but just be mindful that um the way
we produce outputs is not through a forward pass and then we have the output but the way we produce outputs is we
start with something random and then there's a process that obviously involves the model uh that allows us to
step by step by step shape um the outputs. Now
yeah again so this paper is laced with philosophy. We propose viewing thinking as an
optimization procedure with respect to a learned verifier which evaluates the compatibility between an input and
candidate prediction. So
yeah they they they propose viewing thinking like the giant word and concept
of thinking to happen to exactly align with what energy based models are doing
which yeah again I strongly feel there is a degree of reverse engineering
happening right here. All right I promised you no more philosophy and we'll dive into the models themselves.
So what is an energybased model and and what h how do you how do you go about um
how do you go about doing inference in one? So an energy- based model is some
something that works together with uh what we call an energy function. So an energy function typically maybe called e
um has two inputs. So let's call it x and y.
And um the energy function is supposed
to tell you is supposed to give you a high number if X and Y sort of are
compatible in some way. I'm going to draw a heart right here. And it's going to give you a low now wait other way
around. See low energy means they go together.
They're nice. They're close. Uh they just fit just fit right and then a high
number if X let me draw a little lightning and Y they don't like each
other they don't go together. Now you might complain that this is still an
incredibly abstract concept and that is true. Um it obviously depends on what
you want to do. So let's say you want to do next token prediction, right? X would
be um the context, the prompt, whatever you want like the the the the partial
text and Y would be a distribution
over the next token, right? Like a a distribution over vocabulary that
represents what the next token should be. So the energy would be low if that
distribution points only to tokens that are actually you know valid in the
language as continuations for X and it would be high if that distribution is
somehow different. If you were doing image dnoising then um X would be the
noised image and Y would be the dnoised image of the same image. However, if X
is the noised image but Y is a different dnoised image, uh that energy would need
to be high. So the energy function is what we train, right? And that is a a
parameterized function and and that's going to be a transformer in this case.
So we're going to have a transformer model. So the the the the whole model
here is going to be this energy function. We're not going to need an additional model than this. This is the
entirety of the learned parameters. This is different than a GAN. In a GAN, this
would be the discriminator and you would still need to train a generator to work
against the discriminator. In an energy- based model, the energy the energy or at
least in this formulation here, the energy function is all you need and the energy function is what you are
training. So you were training um you you were training a parameterized representation
of a function that and then you feed it with data. You're kind of okay here is here is a sentence the dog blah blah
blah and here is a distribution over next tokens and you train it to be low
if they go together and you train it to be high if they don't go together. You
can do this in various ways right you and we get to that later. One easy way is to do contrastive training. So you
take the same context and you take the distribution the one hot distribution of
the correct next token and then you take a one hot distribution over the incorrect next token and you do a
contrastive training. You say the correct um the correct the the the the
output where these two are where this is the correct next token should be lower
than the output of these two where this is the incorrect next token. However,
that is not a scalable way to go about it and we'll see later. So the energy
function is the the only thing we need. Um so you might object and say hey what's
the difference to loss uh because like a that that just seems very much like like
loss the loss function is also exactly like this right um and the difference is
when you use it. So an an energy function you are supposed to use at
inference time whereas a loss function you're supposed to use at training time. In fact, training the energy function
here, training these parameters itself has a loss function associated
with it. Right? So um the loss function in the contrastive training might
actually be so the loss function might actually be that the energy of x and y1
must be lower or like
minus the energy x and y2 right this would encourage this to go down and this
to go up and that's what we want because this is the correct one and this is the incorrect one. So a loss function is a
is a training objective and an energy function is an inference objective and
that also gives you a hint of what do you actually do with this energy function. So we we only train a model to
predict a single number, right? This thing here, this is going to be a real
number. It's not going to be an output of anything. It's going to be a number. And so what do you do if I only give you
a function that where you can put in the current half sentence and then a
distribution across like across the next. So here is the current half sentence and this here is a distribution
over the next token and I'm simply going to tell you a number and that number is going to be higher if it's bad and lower
if it's good. Well, if you just have that, you're simply going to take the
same half sentence and then you're just going to try a whole bunch of these distributions, right? And you're going
to see, okay, which one is the lowest, right? So you you may be able to one hot
encode all your vocabulary and just slap everyone in and then seeing which one is the lowest. But in fact that's not the
whole space of distribution because distributions can be obviously not just one hot. So you could think of just
slapping in every possible distribution and finding the lowest value right
there. And now we come to the point where you might recognize, hey, this sounds a lot like optimization. And
that's exactly how we go about it. So if you think of it, if you think of it, um
if you have a trained energy function and that trained energy function is such that the if things go together that the
the energy function actually tells you yes, this is low or that's if your energy function is well trained, you're
going to have that. Um so if you do have that then um you can
simply run an optimization procedure at inference time in order to get a good
output. What do I mean? So look at this. This here is um a 2D representation of
the of a uh energy landscape. Again this is not loss. This is at inference time.
So imagine that this axis here is uh this is a bit of a fat.
Imagine that this axis here is um is
your your your uh context, right? This is all the possible context.
No, actually what you can do is if you if Oh, yeah. Okay. This axis here is all the possible
contexts, right? They are discrete. I get it. But we'll just we'll just say
okay, these these possible contexts are continuous. So that's our X and then here are all the possible um
distributions like next token distribution, right? So this distribution right here might be here
and then very, you know, like very spiky distribution or something might be here.
um and so on. And you can see if the energy function is well trained, what we're trying to find are the minima of
the energy function given one of the it one of the inputs is actually our x. So
we fix the x and we change the y and we run an optimization procedure over the
y. So in this case, I've actually done the wrong drawing. So uh both of the
axis right here should obviously this should be y one y like this these are
these are now individual dimensions over your his over your distribution
this distribution space. Now the distribution is obviously um way
way um more highdimensional. In fact it has the dimensionality of the whole vocabulary.
But imagine your vocabulary just has like three different entries. So um
because you need to normalize it, it's a two-dimensional space and and that's what you optimize over here. So you're
trying to do to to find the minima. And how do you do that? Well, by doing gradient descent. So we're doing we're
starting in a random output, a random distribution over the next token. And
then we're doing gradient descent on the energy function back propagating
that to the the input. Right here again we have an x and we have a y like an
estimate of y an initial guess or an intermediate step. We're putting both of
these through a multi-layer transformer that gives us an energy function. And
then we're going to take that energy function. We're going to calculate the
gradient of the energy and we're going to back propagate this through the
transformer to here. So this is going to be um the gradient with respect to yhat
of our energy function where x is fixed right like x is fixed
and we want to know how do we need to change yhat and then we do a little step in that direction and then we reevaluate
it and then we do a little step and we re-evaluate it. Cool. So we're doing gradient descent at inference time.
That's at least one way of doing inference in energy based models. Okay,
we optimize against the energy function. You can see that this has some nice properties like if this is well behaved.
Then if we do some wiggling around here, we can get sort of the variance of
stuff. We can get the uncertainty, right? Is it very wide? Is it very narrow? Um is it very bumpy? And so on.
uh we can also easily rec more easily recognize out of distribution data and
things like this. So lot of excellent properties but the downside is we can't
just get a poof uh an output in a single step. We do have to run this
optimization procedure. All right. So how do you how do you
train this? How do you train a model? Oh by the way energy is not not a
normalized quantity. So energies are always unnormalized uh for scalability
reasons. So all you really know is is it less is it more. Um
yeah. So this would be the in inference procedure. uh we're going to do we're gonna we're gonna sample some initial
guess and then we're going to run gradient descent on the energy function with like some some step size right here
and we're going to return the minimum that we found uh along the way.
Okay, again this is the this is the way we do inference. So how do we train a model?
And there are some challenges right here. And the challenges are that if you just naively train the energy function
to be sort of high on incompatible inputs and low incompatible inputs, you
were going to end up with a very very jagged and a very uh nons smooth energy
landscape, right? This is it's just going to be like okay a lot of and then
wherever your data is and then especially in high dimensions. So therefore what is really important are
some energy landscape regularization techniques to be applied. They have
three of them. Um one is a replay buffer. Uh and that's also often used in
reinforcement learning where you have your trajectories and you sort of keep them around to to grab them to train.
And you you usually do that just to bring some variety into the into the
system and get away from your very very local state and current data. Another thing is um they actually they they add
noise here and it's probably good because I forgot one thing
and that is yeah how how do you train and the trick here is that
you train considering the way you do inference.
Okay, there are two ways to train um or this paper says okay there are two ways to train these things. One is
contrastive. We already looked at that. What is the other one? Well, the other one is saying hey my inference my
ultimate y is going to be um my ultimate y is going to be y0 minus
alpha gradient of energy of x y 0. This
is just a single step right I've done a single step of gradient descent optimization
but you can see well this is my output right and and therefore if I now define
a loss function on my output right and so I define a loss function on y and y
what's the correct one the correct one y from my from my data set right so this
is the distribution over the next token here and this is the actual next token
one hot encoded. I can define a cross entropy loss and I can use this here
as the this is effectively f of
what is it f ofx I guess y equals f ofx yeah so so
if if you didn't know how this came about what would you do you would simply say okay let me let me derive The
gradient right here f has parameters. Let me derive the gradient of the parameters of the loss of f of x and y y
of the correct label and you would back propagate into f and you would back propagate through uh to
the parameters of f. You can do that here. Here are some parameters. This is
a completely linear operation. This here is a completely linear I'm not sure
derable operation. And so what you end up doing is you end up actually backropagating
through your optimization steps. So you're going to backpropagate through an operation which already has a
gradient computation inside. And you know what that means? That means you actually need second order derivatives
right here. However, the second order derivatives aren't too bad um because
you can do uh in this in this case so you require so importantly this loss is
back propagated through the entire optimization process requiring second order derivatives i.e the gradients of
gradients. These are computed efficiently via Hessen vector products
which scale linearly with model size. So it's not the most flop efficient thing
in the world, but it doesn't quadratically explode um if you if you
scale up. So again like this might be a bit weird to people who are really just
used to training forward pass transformer models but we are we're
we're going to train um we're going to train such that the optimate the
inference process itself is considered during the training. So the training
consider the training is okay the loss represents finding a good output including that
inference time gradient descent process. So we train with the inference in mind
and now in to make that scalable we do need to regularize and one part of regularization is to add this noise
right here. So when we do the gradient descent at training time right at training time we're also going to do
this gradient descent. And so we have some sort of energy landscape. We're going to do the gradient descent like
boop boop boop. Okay, going here and then calculating the loss of this and
then back propagating through this inference right here. Um we are going to
also add noise to every one of those steps. And the reason is this helps
generalization and this helps smoothness. So if you are here, let's say this is a top view and your
optimization path looks like this. What you really want is you want by by doing
noise, you're sort of washing this out a little bit, right? And so instead of
treading a path that is sort of really thin right here and the rest, you know,
here and here is undefined, you want to you want to make that path bigger. You
want to sort of broaden the landscape where you reach during training and by
that you make the landscape smoother. You do sacrifice a bit of accuracy
obviously for this but you make the landscape a lot smoother by adding this
noise during training. And so at inference time when and we're looking
for generalization here. So data that we haven't seen during training time. If at inference time data is close to what
you've seen during training, you are not hopelessly lost because you will still be inside of this sort of more wide band
that you've seen and you'll be able to follow and make something sensible out of that inference time data.
Very old trick to add noise obviously but uh very effective and they do this
here as well. The other one is um by randomizing the gradient step size and
the number of optimization steps significantly improved generalization. Again the you don't always want to do
exactly five steps with exactly the same step size. If you vary things up a bit
um then you you can obviously gain a lot. And uh even additionally here
because we are putting in compute at inference time because we're doing multiple forward passes if I already
train doing sometimes less sometimes more
optimization steps I will end up with a model that is much more accustomed to
sort of giving me giving me good answers for all of these situations. And that
hopefully generalizes to a way where I could then also extrapolate at that inference time put in a lot more steps
than I've done during training time time just because I I've sort of trained the
model to be flexible to how many steps I do and I hope that in itself obviously
generalizes. So those are the training um techniques
that they have right here and then they also introduce their model. So their model is a transformer. Uh so they're
introducing they're combining effectively energy- based modeling with transformers and they're saying okay
energy based models have traditionally encountered difficulties with three
characteristics which are parallelizability, stability and scalability. So energy
based models really bad at this, transformers really good at this. So transformers are
good in all of these three things that the energy based models are bad at and so it seems natural that they go
together. So they uh present EBTS energy based transformers which are transformer
implementations designed for EBMS. Um this is a challenge from an engineering
perspective. Uh especially the sort of decoder only triangular attention
transformers need a lot of considerations so that you don't get information leakage across these
multiple uh inference steps that you do in EBM. So you no longer just do one forward pass, you do multiple. And if
you want to benefit from that parallelizable training and if you want to benefit from sort of doing um
parallel computation with this triangular attention, you have to pay very very close attention to how your
your your data flows. They've implemented all of this and their code is available. So that is very very cool.
Um they're going to research two different things here in the experimental section. One is learning
scalability which is sort of the traditional thing which is how quickly can models fit the pre-training data and
the other one is what they call thinking scalability. This is effectively um can
we determine whether model performance improves with increased thinking and by that they mean increased number of
forward passes at inference time. So if we put in more compute can we get sort
of better uh and can we get better in a more scalable in a more rapid way than
other models. So the first thing is they compare with this with transformer plus+
that's a sort of a training recipe to train um
next token prediction single forward pass transformers and you can see right here from these graphs that indeed while
the energy based um transformer does start out on a bit of a disadvantage it
quickly gains over the classic transformer as you for example scale up
training training data, scale up batch size, and scale up the the depth.
Again, these these models like what we're what we're doing like what they're doing is they're effectively showing
like look, the trends are really good. Some of these trends aren't that, you
know, materialized yet. Like you would need to extrapolate somewhere down here
to actually see. And there is still the absolute possibility that at large scale none of this trends actually go the way
that they seem to go. Um but still it's it's quite it's quite promising. So this
is uh training scalability where the energy based transformers already sort
of uh scale better. Now keep in mind the xaxxis right here. The x-axxis represent
you know very particular quantities. The fact of the matter is still that in a
regular transformer one forward pass one training step is one forward pass. And in an energybased model, one training
step means you first have to do the inference procedure during the forward pass and then you have to back propagate
through that inference procedure which all in all is is not you know is quite a
bit more br on your GPUs than a single forward pass transformer.
So the x-axis here if they're like okay batch size number of tokens and so on that's all fine. um in the time domain
you'll see this is quite and that's what we have right here. So you can see in terms of training flops there is and
this is a log scale right the energybased transformers are significantly
away from the classic transformer. However they scale faster. What they
mean is that this slope right here is ever so slightly and you can also see
that right here but this is embedding dimension. Let's stay with flops. The slope here is ever so slightly steeper
than the slope here. And uh therefore if this trend continues there's actually a
future where um because energy based models achieve better perplexity.
you know, the additional flops sort of cancel out and and the um you would need
to invest a lot more training flops into classic transformers than into energy based transformers because the energy
based transformers are just so better at at sort of taking in them those making
use of those flops. So not at this scale but conceivably if you believe the trends and you extrapolate um then that
will at some point cross. So the second part is thinking uh the
thinking so at inference time can we put in some more work and their answer here
is yes indeed. So you can see while the classic transformer obviously does not
scale with number of forward passes, it's it's going to for the same input, it's going to give you the same output
no matter how many forward passes you do. The energybased transformer starts
out weaker, but then as you increase the forward passes, it uh obviously gets
stronger. And that's not a not a surprise because you do start out with something completely random, right? And
then after one forward pass, you've done sort of one inference step, one gradient descent step in the energy landscape.
And so you do need to do a couple to um to to get ahead. Um and yeah, they do
end up ahead right here with a gap to the classic transformer.
Another thing they can do is they can actually look at the energies uh sort of across thinking steps. So how do the
energies evolve? And they see they see one thing and that is that different tokens um have different sort of um
energies. You can see here the light colors represent sort of lower energies. And you can see that throughout the
inference of a sentence uh you do get significantly lower energies at tokens
where they say okay it's it's a lot more clear it's a lot more easy. Um also here
you can see at easy words so to say energy being lower and that represents a
degree of sort of self assessment of these models on and and also an opportunity maybe for us to put in less
energy on these steps. So this ability to put in different amounts of energy, different amounts of flops into um the
inference procedure combined with the fact that the energy function itself can
tell you something about the current state of things and about the uncertainty and about the easiness could
give rise potentially in the future to a very dynamic inference procedure where you don't always have to do oh we always
do a 100 steps or something like this, right? It's it's a little bit the same idea as
the sort of speculative decoding and things like this where oh because you
can you you know something more um you can maybe save some computation. What I
find interesting is that the remarkable thing that there seems to be not a whole lot of difference beyond step beyond
iteration one. So obviously at iteration zero the energies are you know something
very high right but then after iteration one you sort of seem to be in the minimum already and the further
iterations they don't seem to do that much anymore. This is I think confirmed by this plot right here where you do
make the most gain at the beginning. Then again this is this is uh very common in optimization.
Um yeah, not much more other than sort of more of the same. Uh I don't want to go
too much into this. They do video prediction as well and so on. Um and um
compared to what is that diffusion transformers uh I hope you get the idea of what this
is. The scaling trends look promising. I can say that. But obviously again they
because of resource constraints um have not tried at larger scales and
the base case itself is such that you do need just to expand like your fixed cost
to work with energy based models is a lot higher. However, it could in fact be
that at large scales that fixed cost is amortized by the gains that you make and
it could actually be more beneficial to go with energy- based models than with sort of classic models. In all of this,
I find the the paper very cool. Uh, but I do feel like they they bring a lot of philosophy in it and they compare with
models that are not necessarily comparable. Like I don't think chain of thought thinking or or or reasoning
models or anything like this have anything to do with this unless you say oh well they also do
multiple steps and so on but that's to me very abstract to me in an energy
based model this multi-forward pass optimization is just the way you do
inference and you can view that as one inference step and then once you have
that you can might as well do chain of thought with it. You might as well do um
reasoning with that. You might as well train that with reinforcement learning. Right? So these things to me have sort
of not much to do with each other. Um the energy based models have nice
properties no matter no matter what. Right? Okay. I don't want to uh go and
keep you here for longer than necessary. Please give the paper a read if you are
interested. A lot of thanks to the authors. We did discuss this in our
discord paper discussions and actually um the the lead author here was part of
those discussions and we're obviously super thankful for that. That is very cool. If you are interested, come join
our discord. Uh we have a lot of paper discussions all the time and if even if
not, I'll see you around. Bye-bye.


Hello. Today we're taking a look at on the biology of a large language model which is a sort of paper published by
anthropic on their kind of series on transformer circuits where they
investigate what's going on inside of transformer language models and sort of
how do they come to their conclusions. Can we say anything about what's happening internally? Obviously a lot of
people uh nowadays are talking about reasoning models and things like this.
Um but and now like setting all of those aside even basic language models are
able to do quite remarkable things and the question is how does how does that
come to be? Nobody ever programmed any of these models to do for example poetry
or addition um or multilinguality or
anything like this. And the question is can we say anything about how that happens
internally. This is as it the title says kind of a biological problem meaning
that in the past we built machine learning models and we built them in a
way so that they would do something. For example, support vector machines or things like that. We understood quite
well how they did what they did. However, in this new era of large
language models, we don't. we simply train the thing and then um capabilities
emerge if you will. And so we're much more in this realm of being a biologist
and sort of poking the thing and like and seeing what happens and uh by doing
that figuring out what happens. So Anthropic has invested resources here
into looking inside of these models and trying to come up with methods to
decipher what's going on. this paper, I'm just gonna let's call it a blog
post. This blog post right here um goes into sort of the phenomenology of that
and does so by example. So they have a quite a bunch of examples right here.
And for all of these examples, they're able to use a method um and we're going
to see that in a bit what that method is uh called circuit tracing in order to
try to explain what's going on in terms of what features are activated inside of these models and therefore a little bit
and sort of how these models think quote unquote or come to their conclusions.
Now, I have to say for a lot of these examples, um I'm not I'm not so on board
with what anthropics says about it or you know what how they frame things.
They frame things very much in a in an pro-anthropic way. Uh whereas I think in
a lot of these cases simply the explanation is something like fine-tuning works or something like
this. But it's framed in this very oh the internal circuits and boo and the
analogies and blah blah blah but we'll go into that. There is a companion uh
paper/blog post however you will that explains circuit tracing in more depth
and that is um a method by which you can
um trace the circuitry of these models. The way it works, just briefly because
all of the examples we're going to look at are built upon circuit tracing. The way it works is you take a transformer
language model and you train a replacement model. So you train a model
that's supposed to mimic the transformer in um a lot of ways and we're going to
look at that in quite a bit. And then you use that replacement model which is a more interpret
interpretable model. So the as you run data through the replacement model it gives you more interpretable sort of
intermediate signals and you can look at those in order to figure out what's going on.
Um so these are yeah these are called replacement models and they're built
using uh a type of model called a transcoder and that if you replace the
model with a transcoder and run data through it then you get these things they call attribution graphs.
Attribution graphs are sort of like uh which the output is um made by which
features which individual features that exist and these features in turn are
made by which other features and so on. It's going to be easier once we actually look into it. But um just briefly how
how is this transcoder this cross layer transcoder built? So consider a regular
transformer. Um we're going to focus in particular on the multilayer perceptron
features of the transformers. So the attention we're going to leave in place.
We're going to look at the MLP features here. And what we're going to do is we're
going to train this replacement model on the on the right hand side this uh cross layer transcoder to match the output of
every single layer of the transformer. So layer by layer we train a different
model and we run the same data through it and then the
transcoder is just trained at every layer to match the output of the
transformer. So meaning you could if if this works well you could just swap out
the transcoder still use the same attention computation right forward propagating the signal because the
signal at each layer matches the transformer. However, the transcoder you
program with sort of couple of extra things. First of all, and that's just visually seen right here. Uh it has a
lot more um nodes if you will. They call this features. Now, in the MLP, they
call it neurons. Uh in the transcoder, they call it features. And that's more of a uh s like a link a distinction they
choose to make. But the reason they choose to make it is because the transcoder's features are supposed to be
more interpretable. How come? Uh first of all, every single layer of the um
transcoder, it it gets all the outputs of the all the previous layers. So
whereas an MLP only ever gets a signal from a previous layer, so you propagate
the signal um from this layer to the next layer, then it's processed, you propagate it to the next layer and so
on. A transcoder on the other hand at layer five it will get the output of all
of the layers before not just not just like a single
um a single like the single previous layer but all of the layers before. What
does that mean? It means that if some
layer 5 computation needs signal from a layer 2 computation just because it does
so the transcoder can get it directly to that model whereas a transformer would
sort of have to propagate that through the intermediate computations. So that's
one thing is it just makes it much clearer what's coming from where right
um obviously if you train these models well then the models will sort of choose
to go the path of least resistance and just pass the feature directly.
So it means you get a much better insight what information comes from where as you look at a particular layer.
Whereas in a standard transformer again even if it is true that the layer 5 computation needs needs data from layer
2 you wouldn't notice because it would come by means of being put through the
layer three computation through the layer 4 computation that the model has to kind of learn to preserve that while
also doing its other computations. So it makes it much more clear. Second, you train it with a sparcity um penalty or a
yeah like a a sparity regularizer meaning that um
one particular feature or it it makes these features more independent. It
encourages the model um to not use overlapping representations. So usually
in transformers it's very hard to interpret intermediate signals because
um all kinds of overlapping information is put onto it like if you have vectors
representing intermediate concepts or superpositions of concepts and so on.
All of these things are happening. Whereas with the transcoder you encourage it to use one particular
dimension or one particular feature here only for like one particular thing and
uh if it is active then the model is encouraged to not activate the other
features. That's what a sparity penalty does. It's essentially says try to activate as few things at a time as
possible. And that means if the model can choose between propagating a superposition of things where all the
things are active, um it will rather it will rather choose to um layer the
things it wants to superposition into individual features and then deactivate the rest. I hope that kind of makes
sense. That's just sort of more basic loss shaping. But as a result, we do get a model that is trained to match the
output at each layer of a transformer while also uh being sparse um and due to
its trans layer properties and also some few more finer details such as um which
nonlinearities are used and where and in what way. we do get a rather sort of uh
interpretable interpretable output of any data we put through it. Um yeah,
because of these more finer details, in addition to being sparse, uh feature
contributions are also encouraged to be rather linear, meaning that um at the
end we get a graph that essentially says, okay, this feature here is the result of this feature here plus this
feature here, and these together in a rather linear fashion contribute to that
feature. Now obviously what's the downside? Why don't we do this all the time? Why don't we just train
transcoders? As far as I understand uh they are computationally um more burdensome uh to train less stable if
you wanted to train them from scratch like we just train them to match the transformers and also you do lose
performance right if you don't process features layer by layer and rather shortcut and encourage that you may yeah
in turn lose like every regularizer you introduce. such as matching layer by layer, such as sparsity, such as more
linearity losses you performance. And therefore the biggest criticism you
could uh levy here and that's probably the crux of the paper are what the
transcoders do actually the thing that happens in the transformers or are you
simply kind of getting the same output but the transcoder is leading you to a quite wrong conclusion about what
happens and the way that anthropic tries to kind of battle that is with so-called
intervention. ion experiment. Uh so just here um we have the the losses. So we
the trans the trans crosslayerism is uh used with a um done with a jump relu.
You can see that here the output of one layer is not only achieved by the last layer but actually by the um by all of
the layers before and then we have a loss to match the output at each layer
and a sparity
penalty. Yeah. So um you can see this we do this across layers u and just replace
the original transformer model with the replacement model and by that we get
these attribution graphs and the attribution graphs I'm sure they show them somewhere here. The attribution
graphs are just you see which of the features become activated which because the model is now sparse should be not as
many and then what you can do is you can uh sort of group them together um
features that are very similar or or very correlated you can group them together. Additionally,
uh, Anthropic also manually goes in and groups features together and sort of
gives them names and so they get this kind of interpretable thing. So the attribution graphs are
going to look something like this. Um, so here the prompt is uh the National
Digital Analytics Group and then bracket open and then uppercase N. So this is
very uh likely that we're going to look for the acronym of this organization
right here. So the question is what encourages the model to output D A
uppercase as the next token or tokens and the attribution graph can give you
sort of hints of that. In this case, we're looking for rather um linguistic
features, whereas in future examples, we're also going to see that we get more semantic, if you will, features. So, the
way to read these here is at the bottom you have the input prompt. Um, and then we're going to wonder how does this next
token come to be. It's important that what they do is they use the transformer to actually run the inference and then
they use the replacement model to run the same data through and explain what
um what is happening at each of the layers. So this the way to read this here is um
these stacked boxes here are either manually or semi-automatically
aggregated features that are active um at a given token. So the token uh
digital uh activates a feature that the they
call digital. So this is anthropic gives it a label called digital and they interpret this feature by looking at uh
these visualizations right here. So this feature group you can see are all features that are highly active whenever
the word digital appears in a piece of text. So
these here are the highest activations on a given reference corpus uh
presumably similar to the training set of the model and you can see the the
highlighted portion are which tokens are activating this feature a a lot. So here
relatively straightforward whenever the word digital appears these features seem
to be activated quite a bit. You can also see for each feature the top token
predictions um here that that would arise from these features. So
um yeah so digital has obviously different meanings. So you can also see that the
different features even though they react to different words are often already sort of separated by themselves
uh into the different meanings of these words. Um this here seems to react to the uppercase digital more. And then
this feature here um in turn activates
downstream features that uh here that
um cause the model to do certain things. So whereas this feature here is activated for the token itself whenever
digital appears that feature is like lights up the we have to say okay how
does the model actually decide on the next token and deciding on the next token is very often
done as as far as these attribution graph goes by activating features that
causes the model to output something else. So whereas here this feature is
simply saying the model internally recognizes the word digital is you know being in in the
context the feature say DAG is simply a
feature that is activated whenever the model is about is is um whenever the
letters D A are the next output. So you can see the types of uh reference data
activations here. It's all super this feature is extremely active whenever the
next output is DAG. And you can see the top token prediction is also DAG. So
there appear to be what they call input features which is just kind of like features that react to things being
present in the input. And there appear to be output features which is just
features that are being activated uh be to cause the uh output tokens to
be a certain way and then obviously the in between is the thinking so to say. Uh
now this here there is only one let's call it a semantic feature activated. So
by means of opening a parenthesis and starting here with the letter N that
activates a feature that Anthropic calls say or continue an acronym. So what they
do is they go to these features and they look at the top activations and they say ah okay this feature is very active
always on the like first part of an acronym and so it probably causes the
model to continue uh saying an acronym. And so now you can see how this comes together. And sorry I
don't have my usual pen here and uh drawings. I hope that's okay. Um this is
a very interactive blog post so I thought we'll do it with the mouse. So now you can see the fact that so we're
pairing um the fact that the word digital is in
the context and the feature that says oh we're about to like say or continue an
acronym and that those two together are responsible for activating a feature
that is co that is causing the model to output something with an uppercase D.
So all of these features are that so the the model is is deliberately choosing to
activate features to to which would cause a D to be output. Same with an A,
same with a G. And then these in turn cause other features which cause the model to activate other things and so
on. And that's how DAG is produced. So I hope that's clear how to read these. Again the stacking itself is done by
anthropic. uh they are looking at the different features and they are sort of grouping them together and giving them
names according to what they see in the activation and prediction analysis here.
All right. So now we can dive into a couple of these examples. Uh we're not going to dive into all of them but I
hope some of them will make the will make it very clear what's happening. So
multi-step reasoning. Um here they have the
the prompt is fact the capital of the state containing Dallas is and uh the
correct output is Austin. So the the point here
is in order to solve this you need to first determine what the capital like
what state is containing Dallas. And once you know that you need to determine
what the capital of that state is. So it's not it's not a straightup fact.
It's two facts combined together uh that make the output. And so that's an an
example of multi-step reasoning. Uh by the way you can also look here at the the detailed graph of these features and
and explore all of these things together. Uh it's very interesting but we don't want to go too much into that.
So um the question is does the model
internally do this sort of two-step thing where it's oh I first need to
determine what the cap what state this is and then what the capital of this state is or is there some other
mechanism and the way this um attribution graph is structured you can see that uh we have features we have
kind of input features recognizing sort of capital state uh Dallas and so on.
The Dallas in turn is activating another feature that um is Texas.
So this feature uh you can see is both activated whenever Texas is the next
token. So, this would be more like a say Texas type feature, but it's also active
for any Texas related things. So, um you can see Dallas, Fort Worth. I'm not sure
if Georgetown is in is in Texas. Uh but it's kind of like activated for Texas
related things. Um same here. Let's see if we find a feature that actually just
activates on the word Texas, but maybe not.
So the Dallas doesn't activate the same feature as as Texas itself but um you
can see that sort of an intermediary feature is triggered or a group of
features is triggered that kind of like um represents Texas from a multitude of
angles. So that you can say the model internally by a in you know the at the
token Dallas it's sort of internally thinking of Texas and that's mostly caused by the feature Dallas. Now what's
not so nice is that here what what's not shown is all the intermediate features of
um yeah here Texas related all the intermediate features that contribute to
it obviously uh state you know state contributes to it thinking of Texas and
so on um Dallas causes it a lot to think of
Texas but you don't see this because these are heavily pruned Um but just
keep in mind there are always some connections from from these other things always going into into this as well.
So from capital and state uh we activate
a feature called say a capital this is a feature that or a set of features that's
very active before capitals are mentioned right and combined the say a
capital and the Texas related things feature is then activating a feature
called say Austin and that in turn is act is causing the output put token to
be Austin. Now this seems to be quite good reasoning for yes in fact the
models internally do realize or recognize or or in some way materialize
this intermediate step of reasoning. However, what you can also see is that
for example, Dallas is causing relatively in a more direct way uh a the
feature say Austin, right? And uh Texas is causing in a more direct way uh the
word Austin as well. So what you can see is that there's there appear to be an
overlap. So there appears to be at least some sort of internal materialization of
the intermediate fact going on. But there also seem to be quite a lot of shortcut connections where it's just
like oh oh you said oh Texas uh Austin that that's just like word association.
The same thing for Dallas and say Austin like the amount of text where just
Dallas and Austin are mentioned together are uh enormous on the internet and
therefore um you can see that there is also a very direct path from these
features. So this just alludes to the fact that yes, probably these models do learn in
some way to do this kind of reasoning if you will or at least internally
materialize more abstract features. But then the final outcome is sort of an
overlap, a mixture of all of that combined with very shortcut type features where it's just word
associations going out. Um there and here you can see how things like
hallucinations and so on come to be that is whenever these two things are in conflict. So whenever the statistical
word associations and so on are in conflict with the reasoning approaches
and of when they go the same way you get this right here where that's a pretty
clear all the features point in the same way but when they point in different ways that's where you get the problem.
So un like if the model is supposed to output something that is unlikely given
the more surface level statistical associations between the tokens or
between the phrases. They do these um intervention
experiments and this is kind of their biggest claims to why their transcoder
uh replacement model is valid. That's because I say well if it is valid what
we can do is we can actually kind of suppress certain features from
propagating or even invert their signal and then the outcome should be kind of
interpretable. So in this case when they suppress the say a capital feature then
the output is no longer Austin it's Texas um mostly Texas that so the word
the word Texas not Austin anymore if they suppress the feature for Texas then
um Austin is suppressed pretty heavily and so on. So you can see that if they
suppress certain parts of this attribution graph, they can cause the
output to be to be different. Um but it's it has its
limits. Uh but this is kind of like they say well if our method would be kind of
crap then these interventions shouldn't lead to the outputs that they do or to
the change in outputs that they do. I don't fully agree but for for some things it's certainly I I see that's
correct. They can also put in alternative features. So they can run a different different data where for
example um the we're we're sub subbing Oakland here for Dallas which causes
another set of features to light up which Anthropic calls the California features which is kind of the analogous
to the Texas features. And then they can take this signal right here and
substitute Texas for uh the California features and you will get kind of
predictable results. So even though your original tokens are capital state Dallas
and so on, if you substitute in at this particular point the California features, you swap the output from
Austin to Sacramento. If you swap in the Georgia features, you get Atlanta and so
on. Uh, interestingly, they can also do this with other countries or or territories. Uh however they have to do
a much higher um absolute value modification that means kind of like the
concept of a place is itself a bit washy. So the concept of a state
seems to be kind of the same as a concept of a country or a territory or
an empire right here but not exactly the same. So you have to do a higher modification if the thing that you're
substituting in is further away from the the state. So here the the feature
should be kind of like the state of Texas, the state of California, the state of Georgia, which is again and not
exactly substitutable by the territory of British Columbia. just interesting
things they find. Um, which sort of makes sense, I
believe. Uh, and also means that we're still in this kind of superposition realm right
here. All right, the next thing is poems. So, how do language models create
poetry? Specifically, how do they plan out rhymes? So, the there's two
possibilities. either pure improvisation. So they could just go and
towards the end of a of a line sort of look for a word that is rhyming with the
last line or they could plan it out. Meaning they could have even at the beginning of a new line they could have
the end in mind and then work towards achie like towards that. And what we're
going to find is much more of the second than the first. Uh which is quite interesting. So the prompt here is a
rhyming couplet. Um he saw a carrot and had to grab it.
His hunger was and um the model is going to
substitute model is going to substitute which one is it? one of these two a
powerful habit or like a starving rabbit and the point is the point is yeah his
hunger was like a starving rabbit that's what haiku does so the question is this word
rabbit where at what point does the bottle sort of internally already have a
representation of that word in place is it at the beginning of the line or is it
more towards the end when it's just like, "Oh, his hunger was like a starving." And then it's like, "Okay,
I'm going to need some animal that rhymes with grab it." And here we see the attribution
graphs. They focus on particular um token positions, notably the um the the
the it which is the last word, the last token of the last line, and then the new
line character. And it seems like that at the new line character uh certain
features are being activated. So at the new line character combined with the
signal from the last word of the last line character the there's already
features internally representing rhyming with it. Right?
At the new line, the model is already thinking of I need something that rhymes
with eat or it or it or something like that. You can see in the features here
um the uh token here before it um so
it's it activates things that end in like it and then here um activates
things that that end in that great street and so on. Um, so it's already
representing this internally and it even represents two specific words rabbit or
habit. You can see that um these features they uh either they
either represent the word rabbit directly or and this is what we'll maybe
come later too. They also represent the same thing in other languages. They also represent things like bunnies. So this
feature is activated at all of these different things. So it means that not
only does it have the word does it activate the word rabbit but also the concept of rabbit. And that's important
because um if it has the concept of rabbit in mind then it can much better plan out a
like semantically a sentence that works towards saying rabbit right the same
goes for habit. So at the point where the new line is made it kind of first of
all it realizes it needs to rhyme. Second of all, it already internally
represents rhyming with the phonetics of the last line. And lastly, it already
internally has a choice of or a features that represent a discrete uh choice of
potential rhymes for the next line. And those features are then used in
order to when the next token is produced in order to produce like when the next
token at this line is produced in order to produce that. I find that quite cool
and quite remarkable. Not eternally surprising but I do think it's quite
cool. Um intervention experiments also you you can look at them but
um at at this like when you leave the rhyming feature but you just suppress
the features for rabbit and habit at the new line token. It turns out that the um
biggest completion tokens for the last word here are things like crabbit, rat,
savage, and bandit. So things that do actually rhyme, you just kind of
prohibit it to um to to output these particular words
that it had in mind. If you substitute in
um other things for the words that it had in mind, then obviously you can steer it towards different things. And
if you suppress the rhyming or replace the rhyming feature then it will co
consequently do something else like his hunger was like a starving and then um
you simply suppress this feature that says you should rhyme which in turn puts
less signal on these words it has in mind. So now it just goes like a language model is like oh a starving
hunger like a starving something okay jaguar dragon
cobra I yeah again quite interesting and as far as the attribution graphs go
certainly the interventions uh do make sense and I do believe that this gives
an indication of what's going on internally
here interestingly yeah they do find that things like new line tokens but
also in other places like uh end of sentence tokens and so on have a big
influence on on these kind of planning things. So while sentences are being
created uh the models just seem to be kind of language modeling but then at
the end of sentences or at the end of lines that's when the sort of planning and thinking if you will happens and I
believe the reason for it is quite evident because at that point you have you're you're not constrained by you're
not so constrained by language and by grammar and by syntax. um and by having
to continue a particular piece of text as during a sentence. So at the end of a
sentence, you kind of have the maximum freedom to do whatever and therefore you can afford to put all of your
considerations on sort of the more overall planning of the of the text and
not the minutia of keeping grammar right now, which is I think an informative
thing to potentially build more powerful language models. Um, and it's kind of
like a hack that people the the hack of like chain of thought and something like
this is just making this more explicit, right? Like giving the model the freedom
to actually sort of plan out things without having to adhere to the grammar
of the things it's producing right now. Sort of separating those two out.
So they also look at intermediate words like so they looked at the end here um
rabbit but you can also think like okay when it actually has to produce any word during this line is it also is it also
um influenced by this word that it has in mind and the attribution graph clearly says yes. So if here his hunger
was and then a new word is like the word like is being produced. The attribution
graph clearly shows that um the both the
explicit representation of rabbit plus the uh sort of grammar grammatical
features both that um represent obviously the grammar
structure right now but also a feature that says uh we're approaching the end of a rhyming line or active. So the this
word that it has in mind for poetry clearly um represents should I say clearly uh is
clearly in mind of the model. It's hard to not to anthropomorphize as you go through these
things. Keep in mind those are statistical models. And when we say it
thinks of something internally, what we mean is simply that the combination of tokens that go beforehand activate
certain features internally that cause the output distribution to shift. And
all we're saying is essentially that by having a rhyming prompt uh then
internally that causes features to be activated very early on at at like at
the moment it's clear what to rhyme with. It causes features to be activated to already have candidate words already
push their probability up. And so everything else that happens then is
influenced by these features. And yeah, that in in human terms you
would say you have the end in mind and work towards it rather than just going about it and then at the end of you know
when you need to rhyme coming up with something that fits right then and there. So it's an it's an it's an uh
example of planning if you will. Also here the intervention
experiments. I invite you to look at them. But um yeah multilingual circuits
are very interesting. So they wonder how does the model work multilingually. Large language models
tend to be quite proficient in translation in multilingual analysis and
so on. And so the question is what how does it look inside? Does
the model think like does the does the are the thinking circuits specific to a
partic particular languages and there's just some bridges between languages or
is it more like that the internal circuitry is kind of language agnostic so that I don't know a horse in any
language kind of activates the same things and um they investigate this
right here. So the three prompts we're going to look at is the opposite of small is and
then and I don't don't
uh sh something. So they all mean the same
thing and the haiku uh completes the three prompts with correct things.
So, we're going to look at how does this look? And it turns out and um you can
investigate these yourself if you want. It turns out that fairly quickly uh in
the intermediate stages you see these um these features these multilingual
features emerge. So there will be features that are language specific. For
example, uh the word opposite in English and opposite in uh Chinese activate
different features. Not sure if I can even hover. I can't. Well, they forgot this box here.
Um so these are language uh specific but then very quickly you
see language agnostic features arising and um you can see right here.
So the the concept of an an of an antonyym uh being activated and the concept of an
antonym being activated similarly by words like antithesis anthesis an
antithesis uh words like giganz which is the the
German word right here um but also opposition to um I I don't know too many
more languages contrarium which I'm going to
guess I don't know is Latin um but in all kind all kinds of
languages similarly activate these features uh gagan gazett
gagenz and so on it's probably also a um property of the corpus what exactly
is um what exactly is uh these top activations are. But you
can see both features that are activated when the words opposite and antonyym and
antithesis are uttered and also features that are sort of represent the concept
or or the really light up in a lot of um
antonym type situations. And again, these are largely
multilingual. I'm I'm trying to find something else here than just French and uh French and or sorry, German and
English. But I hope you can see that these in turn activate a feature called
say large. So the multilingual feature for anthonym and the multilingual
feature representing uh the word small both give rise to a multilingual feature
called say large like that causes the model to say large in whatever
language. And you can see the top predictions right here are large da big
g uh and so on. And that then is then back combined
with a language indicator features that then causes the model to output the correct
language. So um what we can see is that intermediately there seems to be at
least some sort of an abstraction uh cross languages and then the output
language is then influenced by um individual linguistic features again. So
uh say large large okay where do we have continue in
Chinese after opening quotes? So that's that's simply a feature that um quote open quote in Mandarin which is
I don't know how you open a Mandarin quote but there seems to be features specifically representing the language
of Mandarin and specifically lighting up whenever you open a quote in a Mandarin
uh sentence and that will to me this is more probably like a superposition of a
feature that represents a quote and the feature that represents sense the Mandarin language but it could be that
just the model learned this as one feature and it's um yeah that will then
influence the actual language being output. So what they find here by also going through more through uh these um
substitution experiments uh which do get interesting. So they think of themselves, okay, at what point
do we actually have a crossover um between swapping antonyms for synonym
features and so on in different languages? And the conclusion of all of this, at least to to them, is there
seems to be a sort of an internal like bias towards English. Um, and that could
be because most of the training data is usually English. So that you can say kind of internally there is a bit of a
mix between um language agnostic and and largely English thinking uh and then
there seem to be language specific features that just kind of influence the input and output to these multilingual
circuitries. Yeah. Again you can see that the English graphs uh of of substitute like how strong do they have
to make the intervention to achieve a particular goal. The English graphs always look quite uh different in terms
of of magnitude from the from the
others also here editing the output language. uh you can see that the
intervention strength uh to change the output language to something else than
English it has to be a lot larger than changing something um from another
language.
So although the French here is also pretty pretty
drastic.
Um so yeah again um this is these are the the multilingual features. Now what
I find the last interesting experiments they do here is they do these um
intersection over union features. So they say we collect feature activations on a data set of paragraphs of diverse
range of topics with claw generated translations in French and Chinese. For each paragraph and its translations, we
record the set of features when which activate anywhere in the context. For
each paragraph, pair of language and model layer, we compute the intersection uh divided by the union to
measure the degree of overlap. As a baseline, we compare this with the same intersection over union measurement of
unrelated paragraphs with the same language pairing. So what this means is here they take uh they compare English
and Chinese. They take a lot of sentences that is are translated to both
languages and they run each of the sentences through the model and they see
which features are activated. Now what we're interested in is what's the proportion of features that are um where
that both the translations activate together and compared to the proportion
of features that they activate separately. So essentially meaning how many of the features activated are kind
of this multilingual features, these abstract features versus how many are
more um single language features. And what's interesting is that you can see
clearly that there is this hump in the middle right here. So the hump in the
middle represents the layer. The middle represents the layer depth. meaning that
um most overlap actually occurs in the middle of middle layers which again
concurs with our observation from before that there appear to be input features
more linguistic. Then in the middle there is more the abstract thinking reasoning whatever you want to call it
features and at the end you again have features that are more language specific. So it's like input then the
processing seems to be more abstract and then at the end the output seems to be again more specific to language to
linguistics and so on. I would concur the same graph uh holds in other
contexts as well. Maybe not as easily measurable, but I do recall a lot of papers about you know BERT like models
that have the same conclusion that essentially say the bulk of the kind of
upper level processing happens in the middle section of the model. Um which
also does make sense. Now here they do also claim that the larger model you can see here generalizes better than the
smaller models uh because it has a higher degree of overlap over
um higher degree of multilingual overlap. I don't necessarily agree with
that because you also have to consider that the smaller model is necessarily weaker. Um, and therefore, uh, it's not
really an applesto apples comparison right here. So, it could just be that the generality is exactly the same, but
because the smaller smaller model is a weaker model, it simply doesn't manage to activate the correct features. Uh,
but if you look at which features it activates, they would actually be multilingual features. So, I'm not sure
I can you can make that conclusion from these graphs. they do. But
um yeah, I wouldn't necessarily wouldn't necessarily say that on the terms do
models think in English. They say okay, there is a there's sort of conflicting evidence. It seems to us that claw 35
haiku is using genuinely multilingual features especially in the middle layers. However, there are important
mechanistic ways in which English is privileged. For example, multilingual features have more significant direct
weights to corresponding English output nodes with non-English outputs being more strongly mediated by say X in
language Y features. Moreover, English quote features seem to engage in double inhibitory effect where they suppress
features which themselves suppress uh large in English that's relating to
their prompt but promote large in other languages.
Um yeah, this paints a picture of a multilingual representation in which English is the default output. So there
you have it. Okay, I think there is a lot more to this paper uh is including
like addition which is really interesting. Um but I don't want to keep uh make this video too long. So I
suggest we take a break here. Uh we return next time. I'll try to do that as
fast as possible. And yeah, this paper we've discussed in our Saturday paper discussions on Discord every Saturday,
almost every Saturday in the evening times of Europe. Um, happy to have you join and see you next time. Bye-bye.


hello there today we're going to take a brief look at GSM symbolic understanding of the limitations of mathematical
reasoning in large language models this paper is out of apple which uh continues
to slowly but surely enter into the research area obviously with uh personal
Acquisitions like Sammy Benjo who was previously at Google this was to be
expected but still very cool that more companies are coming out and and doing research in this field uh especially now
that the traditional companies such as open aai and Google are going more out
of the research fields and only releasing technical reports AKA advertisements about their uh new apis
so what does this paper do in short this paper questions whether reasoning is
happening in large language models especially as it pertains to mathematical reasoning they also ask a
little bit the question are the current benchmarks uh notably the GSM 8K Benchmark are the current benchmarks uh
sort of part of the training sets of a lot of these models because some of
their experimental evidence suggests that the models might already have knowledge pre- knowledge of the
questions and lastly they investigate you know how robust are llms to kind of
this mathematical task do they really understand these things or do they just
sort of do pattern matching and the conclusion that this paper comes to is obviously that oh no the llms aren't
reasoning uh they are just quote unquote pattern matching and also there's
probably quite considerable amount of training set poisoning or test set poisoning however you want to call it of
this GSM 8K Benchmark and therefore new data sets new benchmarks are needed and
they provide one of those this GSM SYM symbolic is one additional data set or
methodology of creating synthetic data that's supposed to prevent this um type
of training set conflation or test set conflation all right so a lot of stuff packed into one paper nevertheless it's
quite a short paper and I expect we won't spend too much time the problems I
have with this paper are twofold on one hand I do believe their data set
construction of this GSM symbolic data set is not without problem and we can
discuss that so some of their conclusions I would put like a question mark behind them uh just because of how
they constructed their data set and I don't agree with their kind of assumptions behind that and secondly uh
their whole point about reasoning as such now it's totally fair to provide
research that kind of shows the weaknesses of llms in a given task
and also maybe even why those weaknesses appear as this paper does but then to draw kind of conclusions of oh it's not
reasoning it's just pattern matching and so on without defining reasoning without
defining these terms well is a bit Shady so the question you have to ask yourself
is are humans reasoning just try to and
then you're going to make first make a joke and say oh well some aren't haaha but you know
um if you actually think about it you probably say yes you probably say humans
uh sort of the human brain is has is is reasoning in its assessment of problems
now this is a challenge for this paper because I would say yeah if you gave the tasks here to a
human they would probably fail in a similar way and therefore if this paper
concludes that since the llms failing these ways they aren't reasoning and just pattern matching I don't know I
don't know what does that mean for humans well okay let's cut it down they
are producing a new data set and the data set is synthetic and the data set
is made such that you can kind of produce endless variants of uh the same
problem but with kind of minor changes to explore how robust llms are to these
changes so they're going to take uh a example from GSM 8K which is a human
created data set of mathematical tasks so little high school math questions uh
for this when Sophie watches her nephew she gets out of yada yada this has 31
blocks in it uh animals has eight stuffed animals inside this has nine
multicolored rings so it's it's questions where you need to know the four basic operations like addition
subtraction multiplication and Division in order to reach the answer and it's packed in a little bit of text so you
have to parse out how the different things relate then do the calculations and after that you'll get an answer so
the data set has the question the data set also has uh the kind of solution
process um annotated and then obviously the the final solution here so this is a
good data set for exploring these topics because it's obvously relatively easy to
automatically check whether an llm did the right thing which in the llm world is considerable challenge to check
whether the output is correct or not because you can formulate stuff in many different ways but having you know
mathematical tasks makes this a lot easier so what they're doing is they always going to take a sample of GSM 8K
and then kind of make a template from that so you can see here uh they annotate these kind of things so in this
case it is uh names and entities and so on but then also different numbers so
they give the these all names you can see right here the numbers also get variable names and then they Define how
these things could be filled in so for example a name uh could be any of a list
of names they have internally then family relationship uh could be in this case you so the the the person who kind
of makes the template here um provides these things and essentially
says well what are the what are the things that can be filled into the template you can see especially for the
numbers obviously you can kind of give ranges uh between which the numbers can
be filled in and then you can have conditions so you can the condition uh
is ob one of the conditions is always what is the solution so in this case the
uh solution must fulfill this condition right here um but you know you could
have different conditions so that the problem makes sense or something like this but this is the basic structure of
what they come up with so they take a a sample of GSM 8K make a template out of it that annotate the template with the
sort of valid values and conditions on the template parameters and after that
they have a you know they have a template um and they can generate as
many many variants of this as they want but note it's always kind of the same like the text in between the template
variables is always the same so they're taking I believe uh a thousand or 100 of
these 100 of GSM 8K and make these templates or a thousand and then make
always make um let it somewhere always make 50 I
believe variants of each so you end up with 50 uh data sets essentially so here for
this work we conduct nearly 500 total evaluations a manageable data set size using 100 templates and generating 50
samples per template resulting in 5,000 total templates for each benchmarks
therefore we have 50 data sets of 100 examples each where each example is a
mutation of one of the original 100 samples from GSM 8K so there are 50 data
sets each of these 50 data sets contain the exact same task uh tasks so task
number one is always based on the same GSM 8K template but filled in with different template placeholders and
therefore is kind of a different task now the first question is okay let's
give all of these 50 data sets to the llms and let's see how they
perform and you will be either shocked or not shocked I don't know what you
expect but it turns out that most of the LL M first of all there's two effects
you see that here most of the llms have a quite a variety of variance in their
final performance so you run the 50 data sets for each of the 50 data sets you you um calculate the mean score and then
you plot that in a histogram and you can see that the variance of these things is
huge so uh from a score of 70 to a score of 85 that is that is a much broader
range than the individual llms are a part from each other on the leaderboard
right also you know you can you can see that for a lot of the models now to be said the stronger models so the larger
models for example GPT 40 you can see right here their variance as you can see right here is quite a bit smaller um
than the the smaller models if you will or weaker models however you want to you
want to call it so that's one thing that they point out is that the variance is
really large um but the the kind of bigger model like GPT 40 is smaller
although I do have something to say about that but that's we'll get to that in a bit when we talk also about the
drop in in accuracy yeah oh we'll do that now so the second thing as you can
see the dash line here is always how well they did on the original original data set so on the original GSM so if
you give them the original tasks that they're derived from how well are they doing and you can see for a lot of
models that is quite a bit better than these data set variants so that's the
first indication that where they say hey um there's probably something where uh
where where that the models already knew about the data set right like why else
would it be so much better on one the that one particular variant rather than
all the other variants so they essentially say look our data set is
essentially a distribution of data sets and the original GSM
8K is basically a single draw from GSM symbolic so where would expect that uh
This falls somewhere in the middle but it doesn't it tends to be on the very
right hand side meaning that oddly in this single draw from this distribution
the models perform significantly better than they perform in all the other draws of the distribution except a few
outliers you can you can see like GPT 40 and so on so this is a graph where they show
how much each of the models kind of drops on average compared to their original performance on gsm 8K so you
can see this is gsm 8K to GSM symbolic accuracy drop sorry I having
trouble this is a bar like this um yeah you can see that
the that some models like um GPT 401 mini and so on they have a relatively
small drop in percentage but then you have you larger and larger drops as I
said especially as the models get smaller now I have something to say about this graphic what what they're
making it they're making it seem like um there's a difference in how much uh
these models drop and and that's true but also as you saw before for example
GPT 40's Baseline like GSM 8K Baseline performance is already at 95 whereas for
example um Gemma 2 here is its Baseline
performance is at only 80 something right and then 535 its
Baseline performance is also like at at 87 or something like this mistol or math
strol is at 80 so compare a model uh that is at let's let's do it really
really extremely so let's say model one is like
99% accurate and model 2 is 10% accurate
it right now imagine what a drop by
1% and it's unclear what 1% means in in
this sense but let's just say they drop by 1% point right so this goes from 99
to 98% accuracy and this one goes from uh 10 to
9% accuracy if you look at the other way around the error so this doubles the
error right so the how much error M1 makes is
doubled this one here the error was at 90% And it's now at 91% so this is
barely like a 1% incre this is a 1 to 2% increase in error if you look from that
way the models who already start at a much higher Baseline performance you
kind of have to normalize their accuracy drop if you will uh by their you know
how much error are they making it I think that helps a lot more than looking at what their score is in looking at the
reverse what their error is and then normalizing by that and then I don't know if you can make too much sense out
of this graph if you rescale it by the error they could just all be relatively constant in in how much they drop and
likewise if you rescale this variance here by the Baseline error that the models make then it it you could just
find that well all of them exhibit about the same variance normalized by you know
where they are on the error scale so that's it's kind of um so I just
there's no big Point here I just wanted to point out that graphs like this where you look at relative performance of
models that already start at different starting points can lead you to
different conclusions depending on how much how how you scale
and what your relative comparison point is so keep that in
mind that being said they obviously say hey look since all of these kind of drop
uh it could be that the data set is already the the test data set is already
part of the training data of these models now I have maybe a bit of a
different hypothesis and they are not exclusive so both could be part of this but if you look at how they construct
their data set I want to challenge a bit that they say well our data set is es
essentially a distribution of data set and the original data set is just one draw from this distributions and I would
at least slightly disagree with that why because the original data set was made by humans and humans when they do little
math exercises they will naturally um they will naturally kind of put the
numbers so that they kind of make sense both to each other and in the real world
and also maybe that so that maybe it's a bit nice to compute but especially the
first two right if you say well I don't know the the electricity bill is this high and one you know kilowatt hour of
power costs this much and those numbers when human make these exercises the
numbers that come out kind of make sense in the real world right for example um
in this case right here so uh right
um the the uh the bin of stuffed animals
has X animals inside the Tower of stacking rings has X multicolored rings
on it you can see the Ranger goes range Z go up to 100 so which stacking tower
of rings has 100 rings on it so um I'm
not I'm not hopefully not nitpicking here and maybe this example is just one
example but you can see that especially then if you just sample from these ranges right then the relations between
them are also completely inconsiderate in in these conditions down here you can
end up with questions that would you would kind of be like really why why why
is someone buying 3,000 lers of milk exactly to go with one box of cereal and
um why am I saying this because llms aren't just Mindless calculators llms
are trained on human produced text largely they are trained to act and
predict next tokens in a world of text where that largely is from humans for
humans and largely describes the real world so also they will be more
comfortable in let's say real world circumstances where things quote unquote
make sense so I I hypothesize that at least some at least some of the drop in
performance if you will comes from the fact that their template generated data
isn't from the same distribution as the original GSM 8K but is from a slightly
different distribution where illogical scenarios are as much part of the
distribution as logical well kind of uh World fitting uh scenarios are as much
part of the distribution as world ill-fitting scenarios whereas in the original data set they are naturally
World fitting because humans produce them so I hope that is kind of brings
the point across the other thing is this GSM um symbolic template they are kind
of half done half automated so they're kind of half automated and then kind of checked by humans but also checked by
two Mo like if less than two models pass them then they're checked again and so on so it's kind of a Hal half automated
process to even come up with the data set and then obviously sampling from the
data set is a fully automated process all right let's go on
they do an interesting experiment where they say Hey you know we just kind of sampled all of these placeholders but we
have distinct placeholders some placeholders are numbers right I can change the number of stuffed animals the
number of rings on the tower and so on or I can change the name so instead of Sophie it's it's John instead of
Sophie's nephew it's Sophie's brother or something like this what if if I separate those and I research those
individually you can see that if you only change the uh the
names um then the language models tend
to not drop inaccuracy in fact some of them are even better than their Baseline performance so if you just change the
names right here uh the green bumps then
everything kind of is fine you still have the big relatively big variance
although not as much and everything is is kind of fine however if you change the numbers right see the the blue hump
here then performance drops significantly also here you know either
could be an indication that yes indeed the models have seen the test set in
their training data and they're just kind of recalling the oh this is the one
about the Rings and the stuffed animals and I recall that the answer was 14 and therefore if you change the names they
still get it correct or it could be that you know because if you change the numbers you make some of these samples
kind of illogical it could be could be that
um that that's part of the reason I think something that does support at
least my hypothesis a little bit is that if this was really a function of remembering uh you would like I don't
think the variance would blow up so much you know I don't think the variance would increase you would you would maybe
see the same bump but maybe lower no um although I do have to point out
obviously here you can see the same thing the lower these bumps go the
bigger variance they have and again if you think of just rescaling it by the
Baseline error rate it is not clear that that variance would also increase in that rescaled version in the rescaled
version these bumps would probably or maybe all be relatively the same um but
yeah I would expect I would expect even in this um in this scaling right here if
it was a pure remembering thing you would just take that bump and you would
just shift it out because it's always going to get give the same answer right because it remembers and
then so it it will be kind of the same distribution except
worse no that doesn't make sense because the distribution is how much it got
correct yeah it it could be either um I'm going nowhere with this I'm sorry
but I yeah I hope you can see how how again interpreting these graphs is not as straightforward I find and the paper
just you know chooses one one view here all right then they go
on and say we can now um we can now change the
data set slightly in that we can make this stuff less and more difficult and
by less and more difficult we don't mean oh it's weirder numbers or something like this
um oh so um but they can now change the
questions for example to take away a condition uh or one element of the
question so you have to do less math operations and and consider less things
or they can add conditions to it so here you have an example this is a call from
a phone booth you have to pay this much for each minute of your call the price drop after 10 minutes uh how much would
you pay for a 60-minute call cost so in this sense they could either drop this after 10 minute price drop or they could
introduce a new price drop even later so after 25 minutes uh the price drops even
more and they can do it twice so they can say well after 25 minutes from the start the price drops even more and if
your total bill is more than $1000 you get a 25% discount so the the second uh
bit here is an even new condition so this is kind of minus one condition
easier plus one condition harder plus plus two conditions the hardest so now
the question is how do the models perform if you make stuff easier and if you make stuff harder again let's first
look at the results so predictably um stuff gets worse and what they argued
they kind of argue that well if the models would understand that would would
reason then it shouldn't matter it shouldn't matter how many conditions you put here it's all just you know you just
map them to plus and minus and you'll be fine so if these were actually reasoning models then this drop in uh accuracy
wouldn't happen they again point out that the variances go up as you go to the left but again I feel like rescaling
them by error percentage could just negate that uh outright what I do I find
interesting two things what I do find interesting is that for some models you can see making them making them questions one harder doesn't really
affect them but making them two harder does affect them and that could also be
a property of data set construction so obviously this here you have to put a bit more thought into how you add these
difficulty levels I do believe this particular example here is a good example of how that can kind of quote unquote go wrong or something I feel
like if if you're tasked with introducing a new difficulty you could
if you look at this right you after 10 minutes the price drops by this much per minute you could be just like oh let me
introduce another price drop after more minutes right that's kind of like an added difficulty okay and then someone
asks you well make another difficulty level and then you've kind of like as a human you're kind of like well I've
already done the the price drop again so let me think of something else and then
they introduce the oh if your bill is more than $10 you get a 25% discount so
it could be that just by the way they made this data set um the plus two here
isn't just twice as as hard but is it um encourages the data set makers to
introduce kind of like fundamentally different problems like the here is you
have to do a comparison and a percentage discount versus the the same Concepts that were
already in the question present so in this case you know this is just a continuation from the condition that was
already there so it could be that this plus one + two thing that they are
trying to do here is not just you know we make it one harder and two harder but
that the two harder is kind of of a different nature often so in in a couple
of these questions leading to the fact that for some of the models making them one harder
drops not at all or yeah not at all and making them too harder does in fact drop
the accuracy second I want to say that you know in this territory here you are
at the limit of what the regular human can do like you give this to like a
random person on the street like this this bottom down here I'm I'm telling
you they like a lot of them will have will have trouble especially if they
have to do it in their head which the llms they they don't get help right um and especially if you then put them
under a bit of time pressure how many would actually get that uh I don't I
don't think so I don't think that the humans a lot of humans or the humans
would score like 100% here so again the question are humans reasoning because if
you conclude if you conclude this here is a good indication of whether stuff is reasoning because oh just adding
conditions shouldn't really do anything it's because you really just have to map it to and so then I'm sorry but then
humans aren't reasoning um and yeah let's write big papers about oh humans
aren't reasoning I like I don't expect I don't get why we expect the LMS to do that uh
in this case it seems to they're very human it seems to me um yeah
so that that those are my thoughts on this uh and then there's the last experiment where they
introduce uh noops so they make this GSM noop so they just kind of introduce
random facts that they have they have no uh effect on the answer like okay you
pick this many kiwis you pick double the kiwis and then you say well five of the kiwis were a bit smaller than average
and then a lot of the models kind of f on that is say oh well the five of the kiwis were smaller we need to subtract
them from the Sunday total or you five of the kiwis yeah also the 01 or llama
here we need to subtract five from the total number of kiwis because they're a bit
smaller um yeah so they discover that the uh the models they don't do well on
this no update set so they they kind of drop in performance even if they use
different kinds of multi-shot um Chain of Thought So by the way all of these experiments are doing uh eight shot
Chain of Thought as kind of his standard for this data set so they give them examples in the context and here even if
they give them examples of the exact same question with the noops in just
kind of like change the numbers a lot of them don't don't cannot solve the question
even if they have eight demonstrations of how the noop is uh successfully
ignored if you will so the noop here and then the these are these are where they
give explicit examples with noops and sometimes even the same question except
you know different numbers with the noops in so you have eight demonstrations of how to ignore it and
most of the models aren't good at picking that up and and are still kind of failing noticeably not all the models
so some of the models like Gemma 2B if you actually give them a Chain of Thought um examples where they are
explicitly shown how to ignore this particular piece of irrelevant information they can can do so uh but um
not not anything El but that that's kind of like cheating right because you
know um but all the models drop and this is an interesting this is an interesting
bit first of all again if you give this to a human and you put like the
irrelevant information like this stuff a lot of humans would make use of it a lot
of humans would somehow try to get this into their answer they be like what what do I do with this do I need to subtract
it or something like this like a lot of them would now if you demonstrated it to
them eight times how you ignore it then maybe not right but you probably have to
explicitly point out I am ignoring this right here because it doesn't you know
because I bet you I bet you you just give this to a human on the street like
at least half of them will somehow do something with the five of the kiwis that were smaller than uh average like
promise and you come back to the thing like our humans reasoning uh if yes then
you have a problem with the conclusions of this paper uh because most of the humans wouldn't solve the problems um if
no then you know what is reasoning and how do we Define it and why do we even care if llms do it
because they're doing Mak doing the same things as humans and then it's just about what
is it then about expecting llms to be calculators to be super formally deriving computers
that's not the point we have computers we have programming languages we have regexes and parsers and all of that if
you want that so in my mind I don't know people love to complain especially about
this oh it's not reasoning it's just P matching but I don't see the point
another thing interestingly that has been pointed out on by a member of our Discord when we discuss this paper by
the way we have lots of paper discussions on Discord uh everyone is welcome um and every if if you want to
present the paper uh that you find interesting that that's that's the place
um someone actually pointed out that not only did they didn't really show that
llms can't reason cuz they they never Define reason but what they did kind of show is show that llms are really bad at
pattern matching even though their point is kind of oh they only do pattern matching no they don't because even if
you give them eight demonstrations of how to ignore this particular piece of
irrelevant information in their context they can't do it a lot of them just
can't do it and what better demonstration do you need to show you
that llms suck at pattern matching uh so I I don't know I don't know but yeah now
obviously what happened is that their training data makes them want to consider this
extra information and no amount of patterns in their context is is
overriding this in this case but still um they suck a pattern
matching so where does that leave us that leaves us with a bunch of conclusions that the paper has which
again uh exposes a critical flaw in the llms ab ability to genuinely understand
mathematical Concepts and discern relevant information for problem solving like uh yes but
also the limitations of the ability of LM to perform genuine mathematical
reasoning but what about humans like the high variance in LM performance on
different versions of the same question their substantial drop in performance with minor increase in difficulty and
their sensitivity to inconsequential information indicate that now before you continue all of these are properties of
humans as well their reasoning is fragile it may resemble sophisticated
pattern matching more than true logical reasoning right so we remind both GSM 8K
and GSM symbolic include relatively simple great school math questions requiring only basic arithmetic
operations at each step of these models likely to be more pronounced in more challenging mathematical benchmarks
yeah develop AI models capable of more formal reasoning and that that is where I don't know we have formal reasoning
engines why do we need llms to to do that humans suck at it why why do we
need aren't llm supposed to be doing the things that humans are good at but machines were thus far bad at um yeah so
as we strive to create systems with humanlike cognitive abilities or general
intelligence yeah that that's like that's the disconnect right like I don't know why people assume that
o these things are any indication of human like reasoning abil cognitive
abilities because it seems to me the llms are much more like humans demonstrated by this
paper that's what it that's what it seems to me all right so enough of the
rant I read the paper in full they have they do like experiments are good they have full reports in the appendix and
whatnot I I just I just feel their conclusions are a little bit um yeah you
can come to different conclusions based on the same data that's it all right tell me what you think in the comments
I'll see you around bye-bye

Part 2:
hello there yeah it's cold here
um we cannot be dissuaded from reviewing
papers today we're going to look at
token forer rethinking Transformer
scaling with tokenized model parameters
this seems to be a collaboration of MOX
plunk Institute for informatics Google
and picking University this paper
proposes as it says the token forer
which is a modification of the
Transformer architecture that as they
say treats model parameters as tokens
and therefore introduces a new kind of
axis of flexibility in Transformer
scaling and ultimately that's going to
result in an
architecture where you can add
parameters to a already trained
Transformer or to an already trained
model and then kind of just continue
training it at that bigger parameter
count um in my opinion it there there's
like 5% of an idea and then 95% is like
smok and mirrors trying
to ouch things in modern words that have
already existed for a long time and
there're there are fundamentally nothing
new I don't want to be too harsh from
the outset or though I probably just was
but we'll dive into the paper so first
what are they attempting to do they say
look Transformers uh are require the
substantial cost of scaling these models
remains a significant concern and they
depend on a fixed number of parameters
specifically within their linear
projections and when you try to do any
modifications to Transformers then that
typically requires retraining the entire
model from scratch so they introduce
this token former which leverages
attention
um not only for computations among input
tokens like a classic Transformer but
also for interactions between tokens and
model parameters so previously
previously you had two types of stuff in
Transformers you had um you you would
you would chunk up your text into
different tokens and then you had the
attention mechanism where essentially
you do token token interaction so tokens
paying attention to other tokens and so
on and that giving rise to the next
layer or the next representation of
tokens and then you had these kind of
feed forward networks um that would take
each token and push it through and make
it into the next representation you that
separately so actually both of these
contain model parameters and both of
these interactions are uh done by linear
projections so in the feed forward layer
what happens if so token X goes goes in
here the feed forward Network would have
parameters W you and if you have y here
as the output well that's not the
correct one let's say um X and and X
Prime up here so you would have X Prime
equal to WX or something like this like
a linear linear projection here uh in
the attention mechanism you do have
token token interactions but before that
you actually so a a mention typically
consists of some kind of outter product
of uh queries and
keys and that goes through some sort of
softmax operation and is multiplied by
the values and that gives you so this
here would be your attention Matrix and
then that whole thing would give you the
output of the next the next layer um all
the tokens
together so to come up with the q's and
the K's and the v's you would actually
also do x * like WQ x * w k x * WV so
also here you have interactions between
inputs and model parameters so even to
facilitate the attention you still have
interactions with model parameters and
their whole point is well what if I want
to modify what if I want to add to these
parameters well then I essentially you
know my whole thing here changes my
whole dimensionality changes and so on
um and I cannot I cannot just use the
same model again I I have now a bigger
model my whole internals are bigger and
therefore I need to retrain everything
from scratch and that is maybe not so
good maybe we want to not retrain things
from
scratch so there are goal is to to
replace these interactions here and
these interactions here with another
tension mechanism um if you know
anything about attension that you know
that in principle we could just add
tokens here right if it weren't for the
position embeddings so let's assume we
have no position embeddings we could
just add tokens and the exact same
mechanism the exact same Transformer
could process them no problem so the
Transformer itself isn't dependent on
the length of the input sequence and
they extend that to the parameter space
and say well what if if if we use a
tension right here we wouldn't
necessarily be dependent on the size of
this W Matrix uh and therefore we could
just increase it and that allows us to
add more parameters to the
model okay so that's
essentially that um they're going to
they're going to end up with experiments
like this one saying look if I have to
train a Transformer from scratch it's
going to you know if I want to train a
124 million um parameter Transformer is
going to require me some sort of cost of
training and I'm going to reach a
certain perplexity if I then have to
train a bigger one and I have to start
from scratch it is going to require me
quite substantially more training than
if I can just scale up from my previous
size to this one using our technique I
waste almost no um no computation here
in order to get to that next size and so
on so as I already said they're going to
replace in two particular places in the
Transformer uh the the linear
interactions so first of all you can see
here is a classic Transformer and you
have this uh qkv projection that's how
you obtain the queries keys and values
for the attention that is done via
linear projection so what they're going
to do is they're going to place this by
an attention
mechanism um itself so an attention
mechanism is going to give rise to these
things right here not a linear
projection and then the feed forward
Network also is going to be an attention
mechanism so the trainable parameters
are going to be these what they call Key
param and value param in both cases
these are trainable parameters so
they're not directly multiplied with the
signal but they're supplied to the
attention mechanism um and they are
tokens so they the trainable parameters
you can like the key paramet consists
internally of a set of
um of tokens right so there are n tokens
right here and the value params
obviously as well and then the the way
this works is the input here is used as
queries into these um into these keys
right here and that defines an attention
Matrix that Aggregates the
values okay so it essentially means that
for each um query here you're going to
get an output that is a a weighted sum
of whatever the values are weighted by
how well the query matches the key
that's a standard definition of an
attention mechanism M right uh actually
in no point of the attention mechanism
does it require that you have as many
queries as you have keys and values only
the keys and the value numbers need to
match um and then yeah so you
essentially have the freedom of having
as many key and value tokens as you want
as long as you have the same amount of
each you can see that this is just a
function it takes the
input signal as an input and it takes
the trainable parameters as an input and
it gives you a output of in the order in
the size of the input signal so you're
just going to do uh the queries which is
the input signals times the keys which
is the learnable parameters softmax
multiply by the values which is also
learnable parameters and that's going to
be your next layer
representation uh so this you're doing
this instead of a feed forward
network uh with parameters W of
X okay um and the same goes to obtain
like the Q of the actual attention layer
to obtain the K of the actual attention
layer and to obtain the V of the actual
attention layer you're going to do one
of these operations each which gets a
bit meta and a bit confusing but in
order to obtain the k of the attention
layer um you're going to do an attention
mechanism of the input signal with the
key and value parameters so you're going
to do the input signal X um times the
key parameters for coming up with
attentions
K softmax aggregate the value parameters
of
attention k k right ah now I got even
confused so you have separate parameters
for coming up with the K separate
parameters for coming up with the V
separate parameters for coming up with
the Q just as if you were to do of k
equal
w k x right so just as if you had a
linear projection um where you have
separate parameters of coming up with
the keys the queries and the values so I
hope you can imagine something among
that and you might think oh wow that's a
neat idea all right because I can now
just add parameters here and especially
if I zero initialize the one of the two
only one of the two has to be zero
initialized um for example the the keys
I guess uh no let's say let's say the
values are zero initialized right uh you
can aggregate as much as you want the
values will will the new values will
never actually um do anything uh except
you know once they're actually trained
start TR training start changing their
value so you can just on the fly at
parameters right here and you will not
change anything about the
model so that's pretty
good now why do I have a bit of my
gripes with this work so yeah they go
through everything here you can see that
this is a diagram of essentially what
I've just shown now contrary to before
the diagram goes goes uh top down right
here so you can see this here replaces
the traditional attention mechanism and
then this this here replaces the
traditional uh feed forward uh
mechanism and yeah also here you can see
that this essentially acts as queries
this essentially acts as keys this as
values so you have a matching attention
scores uh using soft Max and then a
weighted sum of the values which gives
you the output and if you have old ones
you can add new ones and they say here
somewhere
how you initialize them you initialize
them with zeros
here so we augment this set by appending
new key value parameter tokens as this
um so you concatenate old and new key
tokens all and new value
tokens and this scaling scheme permits
the integration of an arbitary number of
parameters without altering the input or
output Dimensions by initializing the
Keys okay the keys um with zero similar
to Laura our model can perfectly resume
the model state from pre-training phase
without losing the well- Learned
knowledge facilitating easier
convergence and accelerating the overall
scaling process all right so I have
essentially two problems with this paper
one is even if you take it at kind of
face value that this is a new thing this
is novel this is different and so on if
you actually closely look at their
curves that I've shown you then it's
it's kind of odd so first of all uh here
they say what we did is we trained um
one from scratch right and then one
incrementally
now the the one from scratch if I recall
correctly or this could be the the one
further down I think is the more
elaborate curve right
they the one from scratch here is
trained with 300 billion tokens right um
and the other ones are trained and here
it says 15b 30b and 60b what's actually
happening is that the first one here is
also trained with 300 billion tokens and
then and then you add an additional 15
billion tokens or 30 or 60 billion
tokens to get the respective curves
right here so in in order for this
actually to make sense and to give you a
benefit you assume that your comparison
is someone training a classic
transformer for the full duration for
all the sizes right so you you kind of
consider someone saying like Okay I'm
going to train this one and this one and
this one and this
one and compared to someone like this if
you actually want to go stepwise through
all the sizes this is a benefit because
it starts from uh a essentially it
starts from an already trained smaller
version now what I find weird first of
all is that even at the smallest level
like the Baseline they're different like
this this method here already
outperforms for some reason the
classically even though they're both
just initially trained with 300 billion
tokens which I don't know already feels
like a lot of what a lot of papers do is
they will find good hyper parameters for
their model their whatever they're doing
and then they'll just say like we just
use those hyperparameters everywhere
which I think I've read in this paper as
well which obviously is then you
know it's it's good hyper parameters for
you it's good settings for you and what
so okay so they already start um with
you know further down but then what's
interesting is that for the same size
for example these sizes right here uh
this one's better like yes it costs more
to get there but this one's better and
if you actually consider that someone
has actually skipped the lowest one and
just says well I want to train a
Transformer with 354 million parameters
they're going to train it for this you
know for this much
um they're going to go through 300 bill
billion tokens right
now these models here have gone
presumably through 300 billion plus in
the yellow case 60 billion tokens so
through more tokens and they're
worse that that's a bit what I
find suspect here is that also here you
can see this is this went through 300
billion tokens whereas this here
presumably went through 300
billion at this size plus 60 at least
right in order to get here and then it's
still worse like goes through more
tokens and it's still worse um that's a
bit to me yeah suspect also then I don't
know what this um training cost here
actually represents I'm just going to
assume it represents training this
particular model from scratch and not
training the sequence of these models
from scratch uh but I can see that you
know once you're bigger your cost
obviously goes
up um and that's what makes the cost
lower so the only thing here is right
they can say look we have a lower kind
of total cost of getting
there but then they kind of end up at a
worst place which I mean it's fine but I
just wanted to point out that even these
models they start at the number of
tokens that the from scratch models have
as their final
State okay the second thing is um and
that's a bit more crucial in my opinion
their framing of this entire thing so
their framing of oh we're going to
introduce new parameters and the
parameters are tokens and whatnot if you
look at the traditional trans trans
former it's let's just look at the feed
forward feed forward part of a
traditional Transformer I previously
said you have a token you push it
through and it gets you a new one right
so let's call that X let's call that X
Prime and here we have a set W that's
actually not the whole story even the
very first attention is all you need
Transformer actually considered a token
going through an up projection right
which we call let's call that W1 and
then a down projection again let's call
that W2 and that will give you um X
Prime and in the middle there's some
sort of a
nonlinearity okay so the actual thing
was we have x * W1 nonlinearity reu or
something
W2 and that gives you X Prime now you
can see that also here we have a free
parameter let's call that M because
they're calling it n we have a free
parameter we can make this inner
Dimension as large or as little as we
want and the rest of the architecture
isn't affected at all which is one of
the things they say is suddenly possible
with their architecture because they can
add tokens right the second like the
second thing is I can also here start
from scratch like if I just call this if
I just call
W1 K Tilda and I call W2 V Tilda you'll
be able to see that oh what I'm doing is
I'm multiplying X by K
Tilda right whether this is transposed
or not right who cares it's a linear
operation I have some nonlinearity and
then I have V Tilda and then it looks
all of a sudden a whole lot like what we
had there and the same thing applies if
I want to add parameters I can just take
my K Tilda and I can add depending on
whether you consider row or column
multiplication and I can just add a
bunch of zero
columns right like so and as long as I
as long as to V I add the corresponding
zero rows um or not even I just have to
add rows just have to add rows then it
will be exactly the same so I can just
fill in my K lower sized K and then add
zeros and it will give me the exact same
result while increasing my parameter so
also this has long been previously
possible
um like so and I would argue It's
actually an inferior thing because uh
people who have done this in the past
and uh myself included in that uh it's
probably better to not do this but to
use some sort of actual up projection if
you want to use some kind of lower
trained model and transfer it to a a
bigger model uh some sort of like um
orthogonal projection and so on they
have really nice mathematical properties
whereas filling this up with zeros will
kind of give you a scaling issue in some
sense
um so you might want to actually try
that but nevertheless the zero zero
adding has been done the even even
experiments in changing this Dimension
here has been done if you look at the
original Transformer paper they actually
have ablations varying this inner
Dimension while keeping the rest of the
architecture completely the same neither
the original Transformer nor this paper
manages to actually change the
dimensionality
of the actual tokens because in order to
do that then you would actually have to
like there's no way to just add zeros to
something uh you're actually going to
change the forward propagated signal
there's actually a way to do it um if
you look at what they do in the
attention in the uh kind of attention
part here you could just add uh
appropriate zeros and then kind of your
if you go from an X to a to Q right uh
you could if this is your original
Vector you could just add a bunch of
zeros here and if you do that
consistently you essentially your inner
products with the K's would kind of null
out here and you would get the same
inner products down here modulos some
scaling Factor so even there you can add
zeros so the only thing if you actually
compare
it that this paper does new if we write
this side by side is it will multiply X
by their parameter key key
parameter it will then do aggregate the
value parameters and it does a soft Max
here a scaled soft Max instead of a
relu and it like it uses the same
mechanism inside of ATT tension as well
instead of just but also that like you
could just say well instead of obtaining
our Q values by doing WX we could just
obtain our Q values instead by doing
W2 uh nonlinearity W1 of X right have a
bit of a more powerful representation in
order to come up with these intermediate
values and that's essentially it um the
rest that this paper is is just couching
this change in nonlinearity
into the language of tokens token
parameter interactions token token
interactions scaling flexibility and
whatnot um so yeah that is a bit I don't
know like I feel like even though it's
like even that would be possible but
then they don't mention that they they
never and if yeah if you look at the
formula yeah it's they they never
mention that they never say hey look
even the original Transformer paper
essentially did what we're doing right
here but we kind of have a new way of
looking at it that would also be fine
but the only place where they compare to
like anything classic is this um oh we
have this net to
net way of adding parameters to a neural
network and we're going to compare to
that and we're going to be somewhat
better
here and that's it now there is a slight
chance there's a slight chance that even
the authors thought that they have
something super new because you know
once you start thinking in this oh ah
we're going to replace it by a token
attention mechanism and
whatnot yeah you tend to be a bit
um a bit driven into this world by
yourself but who knows
um yeah so all in all like looking at
the technique by itself I actually think
it's is quite okay it's probably a good
way of messing with one part of the
model size like with there is a certain
flexibility here where you can add
parameters
not like it's adding parameters in a
distinct way
that influences one part of the model
it's it's not like you cannot change all
of the architecture you cannot change
representational you cannot add
representational capacity to the forward
signal itself what you can essentially
do is you can add computational capacity
like the complexity of
transforming um a signal from one layer
into the next layer here that you can
change with this method you cannot
change the fundamental carrying capacity
of the forward signal with this method
right here so there it's a way to modify
one aspect of the Transformer
model that has already been present at
least in the feat forward layer in the
very first Transformer paper uh if you
want to argue you can say this paper
extends that also to the computation of
the attention inputs um and that's a a
cool thing and you might want to look at
stuff in this way in order to understand
them better in order to extend your
freedom of
experimentation on the other hand yeah
as I said a lot of this is just word
couching and if you actually write down
what it does you will find that it's
essentially a different
nonlinearity and that's it all right
this was my
uh look through this
paper and I hope I hope that wasn't too
harsh um I yeah again I do think like
the research in itself is is pretty good
and is a good way of thinking about
stuff all right if you have different
opinions please let me know and I'll see
you next time bye-bye


hello there yeah it's cold here
um we cannot be dissuaded from reviewing
papers today we're going to look at
token forer rethinking Transformer
scaling with tokenized model parameters
this seems to be a collaboration of MOX
plunk Institute for informatics Google
and picking University this paper
proposes as it says the token forer
which is a modification of the
Transformer architecture that as they
say treats model parameters as tokens
and therefore introduces a new kind of
axis of flexibility in Transformer
scaling and ultimately that's going to
result in an
architecture where you can add
parameters to a already trained
Transformer or to an already trained
model and then kind of just continue
training it at that bigger parameter
count um in my opinion it there there's
like 5% of an idea and then 95% is like
smok and mirrors trying
to ouch things in modern words that have
already existed for a long time and
there're there are fundamentally nothing
new I don't want to be too harsh from
the outset or though I probably just was
but we'll dive into the paper so first
what are they attempting to do they say
look Transformers uh are require the
substantial cost of scaling these models
remains a significant concern and they
depend on a fixed number of parameters
specifically within their linear
projections and when you try to do any
modifications to Transformers then that
typically requires retraining the entire
model from scratch so they introduce
this token former which leverages
attention
um not only for computations among input
tokens like a classic Transformer but
also for interactions between tokens and
model parameters so previously
previously you had two types of stuff in
Transformers you had um you you would
you would chunk up your text into
different tokens and then you had the
attention mechanism where essentially
you do token token interaction so tokens
paying attention to other tokens and so
on and that giving rise to the next
layer or the next representation of
tokens and then you had these kind of
feed forward networks um that would take
each token and push it through and make
it into the next representation you that
separately so actually both of these
contain model parameters and both of
these interactions are uh done by linear
projections so in the feed forward layer
what happens if so token X goes goes in
here the feed forward Network would have
parameters W you and if you have y here
as the output well that's not the
correct one let's say um X and and X
Prime up here so you would have X Prime
equal to WX or something like this like
a linear linear projection here uh in
the attention mechanism you do have
token token interactions but before that
you actually so a a mention typically
consists of some kind of outter product
of uh queries and
keys and that goes through some sort of
softmax operation and is multiplied by
the values and that gives you so this
here would be your attention Matrix and
then that whole thing would give you the
output of the next the next layer um all
the tokens
together so to come up with the q's and
the K's and the v's you would actually
also do x * like WQ x * w k x * WV so
also here you have interactions between
inputs and model parameters so even to
facilitate the attention you still have
interactions with model parameters and
their whole point is well what if I want
to modify what if I want to add to these
parameters well then I essentially you
know my whole thing here changes my
whole dimensionality changes and so on
um and I cannot I cannot just use the
same model again I I have now a bigger
model my whole internals are bigger and
therefore I need to retrain everything
from scratch and that is maybe not so
good maybe we want to not retrain things
from
scratch so there are goal is to to
replace these interactions here and
these interactions here with another
tension mechanism um if you know
anything about attension that you know
that in principle we could just add
tokens here right if it weren't for the
position embeddings so let's assume we
have no position embeddings we could
just add tokens and the exact same
mechanism the exact same Transformer
could process them no problem so the
Transformer itself isn't dependent on
the length of the input sequence and
they extend that to the parameter space
and say well what if if if we use a
tension right here we wouldn't
necessarily be dependent on the size of
this W Matrix uh and therefore we could
just increase it and that allows us to
add more parameters to the
model okay so that's
essentially that um they're going to
they're going to end up with experiments
like this one saying look if I have to
train a Transformer from scratch it's
going to you know if I want to train a
124 million um parameter Transformer is
going to require me some sort of cost of
training and I'm going to reach a
certain perplexity if I then have to
train a bigger one and I have to start
from scratch it is going to require me
quite substantially more training than
if I can just scale up from my previous
size to this one using our technique I
waste almost no um no computation here
in order to get to that next size and so
on so as I already said they're going to
replace in two particular places in the
Transformer uh the the linear
interactions so first of all you can see
here is a classic Transformer and you
have this uh qkv projection that's how
you obtain the queries keys and values
for the attention that is done via
linear projection so what they're going
to do is they're going to place this by
an attention
mechanism um itself so an attention
mechanism is going to give rise to these
things right here not a linear
projection and then the feed forward
Network also is going to be an attention
mechanism so the trainable parameters
are going to be these what they call Key
param and value param in both cases
these are trainable parameters so
they're not directly multiplied with the
signal but they're supplied to the
attention mechanism um and they are
tokens so they the trainable parameters
you can like the key paramet consists
internally of a set of
um of tokens right so there are n tokens
right here and the value params
obviously as well and then the the way
this works is the input here is used as
queries into these um into these keys
right here and that defines an attention
Matrix that Aggregates the
values okay so it essentially means that
for each um query here you're going to
get an output that is a a weighted sum
of whatever the values are weighted by
how well the query matches the key
that's a standard definition of an
attention mechanism M right uh actually
in no point of the attention mechanism
does it require that you have as many
queries as you have keys and values only
the keys and the value numbers need to
match um and then yeah so you
essentially have the freedom of having
as many key and value tokens as you want
as long as you have the same amount of
each you can see that this is just a
function it takes the
input signal as an input and it takes
the trainable parameters as an input and
it gives you a output of in the order in
the size of the input signal so you're
just going to do uh the queries which is
the input signals times the keys which
is the learnable parameters softmax
multiply by the values which is also
learnable parameters and that's going to
be your next layer
representation uh so this you're doing
this instead of a feed forward
network uh with parameters W of
X okay um and the same goes to obtain
like the Q of the actual attention layer
to obtain the K of the actual attention
layer and to obtain the V of the actual
attention layer you're going to do one
of these operations each which gets a
bit meta and a bit confusing but in
order to obtain the k of the attention
layer um you're going to do an attention
mechanism of the input signal with the
key and value parameters so you're going
to do the input signal X um times the
key parameters for coming up with
attentions
K softmax aggregate the value parameters
of
attention k k right ah now I got even
confused so you have separate parameters
for coming up with the K separate
parameters for coming up with the V
separate parameters for coming up with
the Q just as if you were to do of k
equal
w k x right so just as if you had a
linear projection um where you have
separate parameters of coming up with
the keys the queries and the values so I
hope you can imagine something among
that and you might think oh wow that's a
neat idea all right because I can now
just add parameters here and especially
if I zero initialize the one of the two
only one of the two has to be zero
initialized um for example the the keys
I guess uh no let's say let's say the
values are zero initialized right uh you
can aggregate as much as you want the
values will will the new values will
never actually um do anything uh except
you know once they're actually trained
start TR training start changing their
value so you can just on the fly at
parameters right here and you will not
change anything about the
model so that's pretty
good now why do I have a bit of my
gripes with this work so yeah they go
through everything here you can see that
this is a diagram of essentially what
I've just shown now contrary to before
the diagram goes goes uh top down right
here so you can see this here replaces
the traditional attention mechanism and
then this this here replaces the
traditional uh feed forward uh
mechanism and yeah also here you can see
that this essentially acts as queries
this essentially acts as keys this as
values so you have a matching attention
scores uh using soft Max and then a
weighted sum of the values which gives
you the output and if you have old ones
you can add new ones and they say here
somewhere
how you initialize them you initialize
them with zeros
here so we augment this set by appending
new key value parameter tokens as this
um so you concatenate old and new key
tokens all and new value
tokens and this scaling scheme permits
the integration of an arbitary number of
parameters without altering the input or
output Dimensions by initializing the
Keys okay the keys um with zero similar
to Laura our model can perfectly resume
the model state from pre-training phase
without losing the well- Learned
knowledge facilitating easier
convergence and accelerating the overall
scaling process all right so I have
essentially two problems with this paper
one is even if you take it at kind of
face value that this is a new thing this
is novel this is different and so on if
you actually closely look at their
curves that I've shown you then it's
it's kind of odd so first of all uh here
they say what we did is we trained um
one from scratch right and then one
incrementally
now the the one from scratch if I recall
correctly or this could be the the one
further down I think is the more
elaborate curve right
they the one from scratch here is
trained with 300 billion tokens right um
and the other ones are trained and here
it says 15b 30b and 60b what's actually
happening is that the first one here is
also trained with 300 billion tokens and
then and then you add an additional 15
billion tokens or 30 or 60 billion
tokens to get the respective curves
right here so in in order for this
actually to make sense and to give you a
benefit you assume that your comparison
is someone training a classic
transformer for the full duration for
all the sizes right so you you kind of
consider someone saying like Okay I'm
going to train this one and this one and
this one and this
one and compared to someone like this if
you actually want to go stepwise through
all the sizes this is a benefit because
it starts from uh a essentially it
starts from an already trained smaller
version now what I find weird first of
all is that even at the smallest level
like the Baseline they're different like
this this method here already
outperforms for some reason the
classically even though they're both
just initially trained with 300 billion
tokens which I don't know already feels
like a lot of what a lot of papers do is
they will find good hyper parameters for
their model their whatever they're doing
and then they'll just say like we just
use those hyperparameters everywhere
which I think I've read in this paper as
well which obviously is then you
know it's it's good hyper parameters for
you it's good settings for you and what
so okay so they already start um with
you know further down but then what's
interesting is that for the same size
for example these sizes right here uh
this one's better like yes it costs more
to get there but this one's better and
if you actually consider that someone
has actually skipped the lowest one and
just says well I want to train a
Transformer with 354 million parameters
they're going to train it for this you
know for this much
um they're going to go through 300 bill
billion tokens right
now these models here have gone
presumably through 300 billion plus in
the yellow case 60 billion tokens so
through more tokens and they're
worse that that's a bit what I
find suspect here is that also here you
can see this is this went through 300
billion tokens whereas this here
presumably went through 300
billion at this size plus 60 at least
right in order to get here and then it's
still worse like goes through more
tokens and it's still worse um that's a
bit to me yeah suspect also then I don't
know what this um training cost here
actually represents I'm just going to
assume it represents training this
particular model from scratch and not
training the sequence of these models
from scratch uh but I can see that you
know once you're bigger your cost
obviously goes
up um and that's what makes the cost
lower so the only thing here is right
they can say look we have a lower kind
of total cost of getting
there but then they kind of end up at a
worst place which I mean it's fine but I
just wanted to point out that even these
models they start at the number of
tokens that the from scratch models have
as their final
State okay the second thing is um and
that's a bit more crucial in my opinion
their framing of this entire thing so
their framing of oh we're going to
introduce new parameters and the
parameters are tokens and whatnot if you
look at the traditional trans trans
former it's let's just look at the feed
forward feed forward part of a
traditional Transformer I previously
said you have a token you push it
through and it gets you a new one right
so let's call that X let's call that X
Prime and here we have a set W that's
actually not the whole story even the
very first attention is all you need
Transformer actually considered a token
going through an up projection right
which we call let's call that W1 and
then a down projection again let's call
that W2 and that will give you um X
Prime and in the middle there's some
sort of a
nonlinearity okay so the actual thing
was we have x * W1 nonlinearity reu or
something
W2 and that gives you X Prime now you
can see that also here we have a free
parameter let's call that M because
they're calling it n we have a free
parameter we can make this inner
Dimension as large or as little as we
want and the rest of the architecture
isn't affected at all which is one of
the things they say is suddenly possible
with their architecture because they can
add tokens right the second like the
second thing is I can also here start
from scratch like if I just call this if
I just call
W1 K Tilda and I call W2 V Tilda you'll
be able to see that oh what I'm doing is
I'm multiplying X by K
Tilda right whether this is transposed
or not right who cares it's a linear
operation I have some nonlinearity and
then I have V Tilda and then it looks
all of a sudden a whole lot like what we
had there and the same thing applies if
I want to add parameters I can just take
my K Tilda and I can add depending on
whether you consider row or column
multiplication and I can just add a
bunch of zero
columns right like so and as long as I
as long as to V I add the corresponding
zero rows um or not even I just have to
add rows just have to add rows then it
will be exactly the same so I can just
fill in my K lower sized K and then add
zeros and it will give me the exact same
result while increasing my parameter so
also this has long been previously
possible
um like so and I would argue It's
actually an inferior thing because uh
people who have done this in the past
and uh myself included in that uh it's
probably better to not do this but to
use some sort of actual up projection if
you want to use some kind of lower
trained model and transfer it to a a
bigger model uh some sort of like um
orthogonal projection and so on they
have really nice mathematical properties
whereas filling this up with zeros will
kind of give you a scaling issue in some
sense
um so you might want to actually try
that but nevertheless the zero zero
adding has been done the even even
experiments in changing this Dimension
here has been done if you look at the
original Transformer paper they actually
have ablations varying this inner
Dimension while keeping the rest of the
architecture completely the same neither
the original Transformer nor this paper
manages to actually change the
dimensionality
of the actual tokens because in order to
do that then you would actually have to
like there's no way to just add zeros to
something uh you're actually going to
change the forward propagated signal
there's actually a way to do it um if
you look at what they do in the
attention in the uh kind of attention
part here you could just add uh
appropriate zeros and then kind of your
if you go from an X to a to Q right uh
you could if this is your original
Vector you could just add a bunch of
zeros here and if you do that
consistently you essentially your inner
products with the K's would kind of null
out here and you would get the same
inner products down here modulos some
scaling Factor so even there you can add
zeros so the only thing if you actually
compare
it that this paper does new if we write
this side by side is it will multiply X
by their parameter key key
parameter it will then do aggregate the
value parameters and it does a soft Max
here a scaled soft Max instead of a
relu and it like it uses the same
mechanism inside of ATT tension as well
instead of just but also that like you
could just say well instead of obtaining
our Q values by doing WX we could just
obtain our Q values instead by doing
W2 uh nonlinearity W1 of X right have a
bit of a more powerful representation in
order to come up with these intermediate
values and that's essentially it um the
rest that this paper is is just couching
this change in nonlinearity
into the language of tokens token
parameter interactions token token
interactions scaling flexibility and
whatnot um so yeah that is a bit I don't
know like I feel like even though it's
like even that would be possible but
then they don't mention that they they
never and if yeah if you look at the
formula yeah it's they they never
mention that they never say hey look
even the original Transformer paper
essentially did what we're doing right
here but we kind of have a new way of
looking at it that would also be fine
but the only place where they compare to
like anything classic is this um oh we
have this net to
net way of adding parameters to a neural
network and we're going to compare to
that and we're going to be somewhat
better
here and that's it now there is a slight
chance there's a slight chance that even
the authors thought that they have
something super new because you know
once you start thinking in this oh ah
we're going to replace it by a token
attention mechanism and
whatnot yeah you tend to be a bit
um a bit driven into this world by
yourself but who knows
um yeah so all in all like looking at
the technique by itself I actually think
it's is quite okay it's probably a good
way of messing with one part of the
model size like with there is a certain
flexibility here where you can add
parameters
not like it's adding parameters in a
distinct way
that influences one part of the model
it's it's not like you cannot change all
of the architecture you cannot change
representational you cannot add
representational capacity to the forward
signal itself what you can essentially
do is you can add computational capacity
like the complexity of
transforming um a signal from one layer
into the next layer here that you can
change with this method you cannot
change the fundamental carrying capacity
of the forward signal with this method
right here so there it's a way to modify
one aspect of the Transformer
model that has already been present at
least in the feat forward layer in the
very first Transformer paper uh if you
want to argue you can say this paper
extends that also to the computation of
the attention inputs um and that's a a
cool thing and you might want to look at
stuff in this way in order to understand
them better in order to extend your
freedom of
experimentation on the other hand yeah
as I said a lot of this is just word
couching and if you actually write down
what it does you will find that it's
essentially a different
nonlinearity and that's it all right
this was my
uh look through this
paper and I hope I hope that wasn't too
harsh um I yeah again I do think like
the research in itself is is pretty good
and is a good way of thinking about
stuff all right if you have different
opinions please let me know and I'll see
you next time bye-bye

hello today we're going to look at where
rnn's all we needed this is a
collaboration of Mila and Borealis Ai
and puts into question a couple of these
modern takes on kind of RNN style
parallela trainable models like S4 and
Mamba and things like this so they they
kind of question are they really as
effective like are they really as
necessary as one needs going to end up
with the conclusion or with the
hypothesis that maybe you know a plain
RNN is just as good as any of them if
you treat it in the correct way so this
is what the paper kind of proposes that
all of these very complex and
complicated constructions that go into
S4 in Mamba might be
unnecessary um it will you will see it
will kind of it'll put the idea out
there but then fail to really
demonstrate that that is the case and
provide some extremely weak experimental
evidence for that that being said it is
quite a it's a hypothesis that I think
has some Merit so maybe uh notably yosha
Benjo is on this paper and that's
probably why a lot of people paid extra
attention to it but yeah take it for
what it is which is a hypothesis with
some extremely weak experimental
evidence attached to it that might still
turn out to be true but also might still
turn out to be you know completely
incorrect so there's a question mark
here that's their hedge now this is
going to be a relatively short analysis
of the paper just because there isn't
that much content if you actually dig
into it uh so I hope to give you the
gist of it right here as always read the
paper in full if you are really
interested the point here is they're
saying okay there's recent successes of
you know things like Mamba and so on and
the whole point there is that look if
you have Transformer models and you have
like some some sequence of text then you
have essentially a big attention
computation on top which means that your
memory requirements uh are as you know
if you have a sequence length of n then
the requirements here are N squared in
terms of memory and compute
and that just fundamentally limits the
sequence length you can put into a
Transformer and you have to resort to
tricks of you know extending that
linearizing attention sparsifying and
whatnot recurrent neural networks on the
other hand are naturally you know made
to handle sequences of arbitrary lengths
a recurrent neural network treats a
sequence not like a Transformer all at
once but treats it by having some sort
of of hidden State moving that hidden
State and then Computing an update to
the hidden state with each new input
element so I move the hidden State and
each element gets included into the
hidden State there's some computation
and then that spits out the new hidden
State and so on you can see how you know
this is much more akin to I don't know
very very basic or very very reminiscent
of touring machines and whatnot so
sequential computation over values
meaning that your requirement here of
memory is is just constant um and you
can process arbitary long sequences
however rnns have always suffered from
two downsides uh with respect to you
know being applicable in the same
regimes as Transformers are first they
have to perform what's called back back
propagation through time so back bptt
usually called what that means means is
that if I want to train um let's say I
have a computation here that takes my
input token let's just call these tokens
and is some MLP or some some sort of
feat forward Network that just
transforms it a bit and actually put to
in order to put it into the hidden state
in order to get the gradient for that I
have to over here's my loss I have to
back propagate this all the way through
all of these computations of the these
of these intermediate steps so I have to
back propagate through all of these
computations until I'm here whereas with
attention if I have to lost some here
there's the attention Matrix I can just
sort of go through the attention Matrix
bada Bim I'm here right that that's a
matrix computation I can relatively
easily assign I don't need to go through
all of the computation of all of these
tokens in order to reach that token so
back propagation through time has been
very gnarly for r in the past especially
there tends to be a fun a limit of very
few steps where which you can actually
do that before the gradient degrades
accumulates noise and whatnot so not
very good um and especially in plain
rnns that was a big problem then lstms
and grus were invented that alleviated
that problem somewhat somewhat but it's
still a a um a big problem establishment
of like
and S4 and these things solve that quote
unquote by removing the input dependence
of the computation of uh sorry the
hidden State dependence of the
computation of the um how you take the
input into the hidden State what do I
mean by that so
usually you have your hidden State
coming in here of Step T minus one and
you have your current token XT and the
means by which XT gets included into
here can be influenced by the hidden
state which is very you know that's a
very natural concept for example if you
have a word such as um
bank right that word first of all it
could be a verb or it could be a noun if
it's a noun it can be you know either of
different mean meanings of that word so
the way you include this into the hidden
State you might already want to know
what you know what the the past was so
you might want to have some information
about the past uh of the current meaning
of the current hidden state in order to
pre-process this token to decide what of
this token do I even want to
remember this is very powerful but it
also means that you get this ual
dependence now the gradient has to flow
through not just through this path here
but also through this path right here
meaning that the computation here is
fundamentally intertwined with the past
hidden State these newer State space
models they remove this one right here
they just say well you don't get to look
at the past you just you have to look at
your current input you have to decide
how do you want to include that into the
hidden State and
inclusion then is by done by like a pure
sometimes like a pure plus like you just
get to add something to the hidden State
and that's it and that makes first of
all it makes it extremely efficient to
compute many things in parallel uh here
because you can compute all of these
things like if you think of the next t +
one this computation here has nothing to
do with anything before so you can just
compute it in parallel right you don't
have to wait for the past you can
compute it in parallel and you can train
it in parallel so a very big
advantage and uh yeah you can you can do
all of this in very efficient way which
we're going to look at called The
Parallel scan uh yet you lose that
dependence on the past now there is a
good argument that you can actually make
up for that and you know these states
based models they say well you know
empirically we're doing quite fine so
but that's the idea behind it okay what
this paper does is it
says looking at Mamba looking at S4 and
so on these are very complicated things
can't we like break
down
the the actual essence of the thing to
its fundamentals and then just see
whether that already does the trick
whether that already gives us the
performance of mum and S4 or almost the
performance without having to do all of
that complex stuff that these things do
so what they're going to do is they're
going to try to figure out the minimal
thing that has this property of the um
the hidden State being not influenced by
the computation of the input state so
they're going to construct RNN cells
that have this property and
then run them run some experiments on
them so we're going back to
the lstms and the grus and the lstms and
the grus as we said you look here for
example the lstm actually the the gru is
easier let's just focus on the gru for
the purposes of this video the lstm is
analogous except the lstm is a bit more
complicated in my opinion
unnecessarily uh yeah so the the next
hidden state is computed from
the last hidden
State and a proposal hidden state so you
have the last one you have the proposal
and then you have a trade-off parameter
Z that decides what to forget and what
to include so Z is like a mask and
wherever Z is close to zero you kind of
retain the old hidden State and wherever
Z is close to one you come in with the
new hidden State this gating mechanism
is largely responsible uh for the
success of rnns in the past because
without that you would very quickly have
like an exploding hidden state or a
Vanishing hidden state so this is um the
gating is really
important
but now you can see the problem so both
the Z you can see here is itself
dependent on the last hidden
value and also this proposal hidden
state is dependent
on the last hidden value here not in one
way but in two ways because this R gate
right here is also dependent on the last
hidden value you see very intricate
computation you can imagine the
computational graph that creates and the
back propagation that needs to take
place so how can we uh make this easier
how can we introduce concepts of Mamba
and S4 well let's just say you are not
allowed anymore to look at the past
hidden state for
anything except you know here it is
explicitly sure the current hidden State
can include something in the past hidden
state but this here and this here like
the the trade-off and the proposal they
must be completely independent of the
past hi state so let's just strike that
out strike this out then obviously this
is unnecessary then this whole line
becomes unnecessary and that leads you
with
the the Min what they call the Min Gru
or also one of the one of the
nonlinearities becomes unnecessary
because naturally the hidden state is
already normalized then at this point so
the Min
Gru says okay the next hidden state is
again it's a trade-off between the last
hidden State and a proposal hidden State
The Proposal hidden state is simply a
function a linear function of the
current input and the um tradeoff is
also simply a function of the current
input so you only get to look at the
current input and you need to answer how
much of it do you want to include into
the hidden State and which parts or what
what of it do you want to include and
which part of that do you want to
include and with that you can Implement
what's called a parallel scan a parallel
scan is a extremely fast way of
computing a what's called a prefix sum
or um well sum for plus but it also
works for for other operators so
parallel scan is an algorithm method for
computing n prefix computations from n
sequential data points VI an associative
operator and the point here is that it
needs to have the specific
form of something like this right so
this is one instance where you can apply
a parallel scan if you have this form in
that your next hidden state is the past
hidden State times a number plus a
number where the numbers A and B are
independent of the V like the the v's
they are not allowed the VT minus one
and any before are not allowed to
influence the computation of a and BT
then you can implement this in a very
efficient way in this method called yeah
the the prefix sum or a parallel scan
and there are couple of variants to do
it so if you can do a lot of work in
parallel such on a GPU you have
algorithms that um run in log and time
because you can parallelize but you can
also implement this in a different way
to have a kind of a recursive
computation where the tree has depth log
n by abusing the particular structure
that the problem
has this is the whole trick that like S4
and and Mamba have of saying look we are
an RNN but we can be trained uh in
parallel or we you know we have this uh
we have this parallel inference nature
as Transformers
do now you might say okay you can do
that but you kind of neuter your neural
network work you you just kind of make
it
completely uh less strictly less
powerful than an lstm and that is true
that is true but what people would
probably say you know against that
is yes the let's say you have already
you have a hidden state that comes here
and you want to include this value right
here yes the hidden State uh sorry the
computation T minus one this computation
here is not allowed to look at that one
so it has essentially no notion of the
past if you
will however once you compute that H uh
T um HT itself has a notion of the past
because it includes HT minus one because
HT specifically is HT minus1 time some
number a t plus some number BT
so if you add now a second layer which
is very popular to do you can kind of
make up for this fact so in the second
layer these hidden states are going to
be the X's here and yes this computation
here so let's call this H um of layer 2
T minus one yes this computation here
isn't allowed to look at this hidden
state right here but it is allowed to
look at its input and its input
has a notion of the past looking at HT
minus one and so you can see that after
layer 1 you actually do get a past
dependent um a past dependent
computation even though you kind of
forbid it in the current layer this is a
bit reminiscent of convolutional neural
networks where you can say yes each
layer can only look at the couple of
pixels beside it but if you go a layer
up then the receptive field quote
unquote becomes larger because you can
implicitly look at pixels further away
because the inputs are already have
already that kind of baked in so I hope
you can see the parallels here that
lstms and grus like the classic rnns
they're trying to do a lot of stuff in
one layer that you can just expand over
multiple layers and you kind of get the
same thing however it's a a matter of
trade-offs so this will need there are
computations where this method will need
like n layers where a classic lstm could
do it in a single layer right same with
Transformers this there are computations
where this will need n layers and a an
attention layer could just do it all in
a single layer so that's the
tradeoff now I don't want to go a lot
further into the into the details here
yeah as I said please read it um it is
you know it's interesting but it's
essentially saying we break these
principles down to the Bare Bones so the
Min Gru is like the the the absolute
minimum of a model that still has this
property of sort of being able to
compute in
parallel and now let's look at how it
behaves so first of all here are some
performance characteristics you can see
uh the runtime
um the training runtime for lstms and
grus linear in the sequence length
because you have to forward propagate
the hidden States one by one you
literally cannot do anything on the next
hidden State before you have the last
one because even the input computation
is dependent on it whereas with the Min
variance it's just flat it's just
constant
because it was log in but um looks like
constant it so and notably the Mumble
line hides behind these lines so it's
the same as Mamba uh the memory
requirements are uh also they are uh
larger depending here on the on the
sequence length but they kind of you
know are in between what Mamba uses and
what the the lstm ones would use so um
why might that be uh because you the
more work like you do at once uh the
more your memory grows but what we care
about is simply the
um the the overall trend which in all
cases here is
linear which is exactly what we would
have expected so this is just kind of
validating what we expected now we come
to the actual experiments and this is
kind of the weak part of the paper so
you can see that they their staple here
is the selective copying task where you
have a bunch of you know you have a
sequence and then some of these sequence
elements are marked uh that you have to
copy them to the end and so the the the
Lang the model has to um copy those to
the end so it's kind of a way of testing
can you have long range selective
attention if you will you can see that
the these models they struggle heavily
in one layer but as soon as you add
layers they are now um they're their
performance so this is a testament to
the fact that yes a single layer is a
lot weaker than like a single layer of
an lstm or so however as you add layers
you get back this property of past
dependence if you will of of time
dependence uh time depended input
computation they further test on
reinforcement learning uh benchmarks but
if you know these problems like the half
cheetah and the hopper and so on these
are extremely simple uh problems and um
so we've talked about this paper in our
Saturday paper discussions on our
Discord by the way join if you are
interested uh it's always a fun time
we've we've discussed some people who
know more about these than I do uh we
very confident that these are such easy
problems that you probably don't need
any recurrent net like you don't need
any sequence model in order to handle
them like looking at the single current
state is probably enough and if not
looking at the state and one last state
like two frames is enough to just solve
them like they're so easy so what they
want to say here is like look we're
doing quite well in fact we're doing
probably better than like the S4 in this
and du the transition decision
Transformer is also on par with us but
it's extremely easy benchmarks and
extremely like low space like low
dimensional and so on also here
comparing the selective copy task uh to
the others yeah you can solve it the
others can solve it
too it's it's such an easy Benchmark and
it yeah although what I have to say is
interesting here is that you you can see
that not all the configurations of these
other models can solve it now obviously
the authors here have chosen a benchmark
that their model can solve but it's
interesting because you would expect
like any model to solve these and what
that means to me is that um contrary to
a Transformer which you can just
probably throw at any sequence task and
it will probably do fine because these
models are strictly quote unquote less
powerful um in order to enable their
tradeoffs it is it is probably necessary
to actually apply them to data and
problems where that suits their
particular architecture so the fact that
here some of these cannot even solve
this selective copying task doesn't mean
they're bad it just means that you have
to apply them to problem
and data where their particular
architecture is suited and then they
might do just as well as Transformers
and that's what we see in all these
papers however there are relatively easy
problems for each of them where they
completely fail uh where a Transformer
would probably also quite easily succeed
so you have to think a bit more in order
so there it's kind of like no free lunch
right and and then lastly they do
language modeling on the Shakespeare
data set which is character level small
um language modeling data set saying
like hey look uh Transformers actually
have a kind of a harder time here and uh
we are we're on par with the mambas and
so on or this is mambas
specifically and yes yes Transformers
probably have a harder time character
level language modeling because
character level language modeling is
extremely local and
extremely like predict like that there's
not much integrated reasoning you have
to do because most of the task is just
completing known words from a dictionary
right so a these sequence level
models even if they are input
computation is time independent that is
very suited because they can just make
the hidden State Decay very quickly
which means they just pay attention to
the very recent past a lot and then they
don't need to do this computation very
selectively they can just kind of gear
towards so these input as these time
independent rnns you have to imagine
that what they essentially have to do is
they have to set up a kind of a kernel
that is always the same so they they
have to compute they have to integrate
the current and the L and data in the
same way all the time because they don't
get to do dependent computation so
problems where you can a priori kind of
decide how to include past
information are very suited to them and
character level language modeling is one
of them because you a priori know okay
I'm just probably need to look back at
like the six past characters and then
you know how you have to integrate them
exactly that's a learnable function in
order to complete the current word uh
because most of the loss is just
completing the word and not necessarily
knowing the grammar or even knowing
quote unquote Common Sense knowing the
language in order to do long range
language modeling sorry this is a bit
bit of a rant what I was trying to say
is good point there as good as Mamba on
this but also the fact that Transformers
are a bit weaker right here just means
the problem is very suited to them and
most of all again very small scale data
set at least these other papers have in
part shown that on you know heavy duty
tasks they're on par with the
Transformers like Mamba S4 and so on uh
they are I would say for some problem
serious contenders this paper right here
raises an interesting point in that hey
isn't all of that a bit Superfluous like
could we just break it down to its Bare
Bones and it will work just as well as
these complicated uh State space models
and the hypothesis is valid but nothing
in this paper
actually shows proves or indicates that
because the experimental evidence is
just so so weak in any case I don't
think it's unreasonable to expect that
to expect that you know if you just
scale these things up quite a bit they
will do
might do almost as well as like the
complicated State space models because
the their fundamental properties are the
same and and then it's just maybe a
constant Factor difference which could
matter or could not matter all right
that was it uh let me know what you
think and I'll see you around bye-bye

Part 3:

hello there today we're going to look at xlsm extended long short-term memory by
maximilan Beck corbinian Pepple and a team around them especially the
corresponding author here is Zep hiter who you may remember was the original or
among the original inventors of lstm's long shortterm memory neuron Network
the most popular type of current neural network there was invented in the '90s
obviously where all the good things were invented and this paper is a new take on
an lstm with all the learnings that we now have from the llm world like from
the world of Transformers and so on so the central question here is how how far
can we push uh lstm type architectures or recurrent architectures in general
and how much can they compete with Transformer based models or attention
based models when it comes to specifically language modeling so
obviously I I don't know but if like let's say if let's say there is something to this right then that would
be a huzzah back for the lstm authors uh
so there's also maybe a bit of of personal interest right here but I do
think it's a cool work in a cool Direction and notably it ties into a bigger picture uh around does it even
matter what architecture we have do only parameters matter you know what's really important in these kind of things and we
can get all into all of that a bit later but first note the beauty here the paper
starts with in the 1990s you know you know you're a you're about to read
history when a paper starts but with the word in the 1990s yeah in the 1990s I uh can't
really remember much of that time but some people can and this paper says the
constant error Carousel and gating were introduced as the central ideas of the
long shortterm memory since then lstms have stood the test of time contributed
to uh numerous deep learning success stories in particular they constituted
the first first large language models already were out of the gates with
extraordinary claims there is a I don't know there's something to be said about
lstms they certainly had their impact they had their impact in numerous domains right they they were widespread
their properties make them really good sequence processors even sequences that Transformers struggle with like time
series and I don't know like heart rate data and whatnot audio data all of this
stuff lstms could you know are applied they don't use uh humongous amount of
resources and so on and yes they were also used for language modeling notably
a lot of the um machine translation architectures and so on were powered by
lstms um yeah but I don't know the the
the the 1990s crew must kind of formulate everything that's happening
today as oh this was already done right so now the hype is around large language
models by which we mean specifically language models that take you know they
go really like big in parameters it really achieve their achieve their width
and achieve their performance by scaling right and then pumping as much data as
possible into them and yes you may say that some of the language models back
then and some of the translation models were kind of that but always being like
Oh we we did that we already did that back then we did Transformers back then we did Gans back then I don't want to
get into a fight here okay uh no opinions on those things I'm just saying
whether it's true or not the constant not just referral to hey that was great
back in the day but on top of that being like you know all of this stuff today we we actually did that already back in the
day yeah sorry rant over I do have a bit of an issue with the first large
language model I would I would strongly argue that the first large language
models were very probably the engram models that in particular Google built
on really large scales like the number if you count these as parameters the number of parameters of these were like
through the roof and therefore I would seriously consider the lsms to not
be the first large language models like as long as we're you know measuring
gonads um I would I don't know I'd be interested in in revising that however
the Advent of Transformer technology with parallelizable self attention at its core Mark the dawn of a new era
outpacing stms at scale we now raise a simple question how
far do we get in language modeling when scaling lstms to billions of parameters
and that's a valid question leveraging the latest techniques from Modern llms
but mitigating known limitations of lstms and that is is a really good
question right if we have all these learnings from Transformers some of them
aren't really particular to Transformers some of them are just like you need to normalize your
data in between layers in a smart way like yeah okay that that makes sense
right you need to use multiple heads if you do sort of single token processing
or if you you do processing it it pays off to have multiple heads in an operation that's fair we can do that
without Transformers right and a lot of architecture like State space models and
so on have actually implemented a lot of these things without Transformers and therefore you could say you know could
we apply some of that to lstms also could we kind of mitigate the
limitations that we know lstms have and if we do that what happens if we scale
them up are they going to be as good as Transformers are they going to be better are they going to be worse you know who
knows and the bigger question here is is that if you just have like some function
some trainable function um does it really matter how it's built or is it just the number of parameters that
matters given given that you train it like with enough data and you know well
with learning rates that fit and so on but does it matter or does it not matter
if it doesn't matter I mean lstms they do have constant memory usage and
therefore you know could be a great candidate they can go for infinitely long sequences could be a great
candidate all right they say we introduce exponential gating with appr appropriate normalization and
stabilization techniques secondly we modify the LSM memory structure obtaining two different things first SL
lstm with a scalar memory a scaler update and new memory mixing two M lstm
that is fully parallelizable with a matrix memory and the co-variance update rule y y then they do experiments the
experiments are good I can I can already tell you the their experiments are good and then the entire Community looks at
their experiments and goes like and we don't have code yet so
that's the current state where we're in uh so what they do is they take lstm
here on the left right and they modify it in two different ways like two separate ways of modifying the SL
lstm is um in the most plain sense it switches out some of the nonlinearities
some of the nonlinearities right now are sigmoids it switches them out via exponential function then it handles
some of the stuff that comes as downsides to using the exponential function because it it it grows pretty
quickly and then they also have this thing they call new memory mixing
however that just seems like it's uh it's a like that's just really new if you
look at the really really old old lstm but parts of that have kind of already
been been part of lstm world for many years um at least the way I understand
it secondly they changed the lstm to the mlsm note these are two separate things
right they derive two different cells um recurrent cells and then they make two
different blocks from from those and each block here then serves as a layer
in these stacked um in a in a stacked layer sense so here is depth of the
neural network and they can for example switch it with an M layer then an S layer then an M layer an M an S like
arbitrarily stack these layers on top of one another so whereas the Transformer
would only have like a single block that consists of a tension and and feed forward um this architecture has two
different blocks that it chains together in arbitrary fraction and that that's
how it propagates signal so the M lstm also has the exponential gating so again
switching out nonlinearities and then it has this Matrix memory The Matrix memory
is something we're going to look at notably if you know L lstms they have an internal state that they maintain so
they go over the sequence and they try to extract information from each each
element and put it into an internal State and then the later sequence elements can kind of read from that
internal state which is really different from the attention mechanism right um
what they do is here they extend they ask themselves how can we make this memory more have more
capacity um the naive way would just to be increase in dimensionality however
what they do right here is they have kind of an associative memory usage um
and we'll get to that and that comes with this covariance update rule uh and that then lends
itself to parallel training actually the parallel training is just a result that they kind of drop one connection from
like they drop one hidden to Hidden connection and then it's a it's a parallel and um doesn't seem to hurt so
they can they can just do it all right then they yeah they they G them notably
this m lstm thing is then parallelizable in training like a attention is so we have a lot to get into first maybe it's
um interesting to just see the difference between kind of what un like the the level that we care about what
attention mechanisms or Transformers do and then what kind of lstms do and why
why they have fallen out of fashion so in a Transformer and let's say we're
looking at a decoder only Transformer you have a sequence elements right and
then I I care about forward propagating signal across the sequence think of next
token prediction or something so I want to compute signal for this particular token here and what I do is in each
layer of my neural network I able to kind of look at all of the past using
attention right dynamically look at all of the past and aggregate information from that and then at the next layer
again I'm able to look at all of the past at the same time using the
attention mechanism and grab information dynamically right dynamically aggregate
information that I need now this has an advantage obviously I get to look at
every single piece of information every single time and
therefore that there's almost no better case for you know how can you work with
sequences every element gets to look at every other element in every single step
on the other hand you may also know that this does have considerable cost especially since this aggregation here
is uh consists of nonlinear functions um especially in attention I have I
multiply curries times keys I do the outer product and then I do a a softmax
out of that and then I multiply that by the values and this soft Max right here
is a highly nonlinear operation and therefore that kind
of that that denies me the opportunity to do a lot of optimizations so I have
to almost realize this Matrix in here which is really big or I have to do that
sequentially which costs time in turn an lstm would go over a
sequence one by one so let's say let's say I have I have a sequence um I want
to predict the next token the lstm would already have a hidden State somehow a
hidden State um I'm going to draw that here and
that hidden State already has information from all of the past that it
has accumulated one by one moving over the hidden state so the hidden State
already exists and all I need to do is I need to inte create information from the current element into the hidden
State and produce the next hidden State and from that I then also produce
produce the output of the current of the current step either from here or from here like from somewhere I produced the
output of the current step so each step only looks at itself and the last hidden
state so there's information that gets passed from time step to time step step and there is nothing in this that allows
me to explicitly look back at the past elements now this is good because it
means I just use constant memory right I have this hidden State and it's the same size as this hidden State and I just
need to find a way to update it so constant memory and I I need to learn a smart way to deal with that memory on
the other hand not being able to look at the individual sequence elements means that the what I encode from this state
into the next hit memory that's all there will ever be right like no future
thing can ever look at this element other than look at the information
that's available in the hidden state so it's almost like you need to learn to extract the exact right things that any
future any future element would want to know of you and you need to provide it
in a compact way and you so that it's addressable and you need to make sure
when you update the memory that you don't override anything that's important that's already there because you're
going to use the same bits to represent that you see the challenge that's why
originally people developed attention they said hey our rnns are good but they you know they would benefit from being
able to directly look back like at this element right here and that's how the attention mechanism was born and then
it's eventually people said actually we we don't need this this kind of hidden State thing anymore right now why was
this a benefit to not need the hidden State thing anymore and that's where we need to actually look at how the lstm is
built and that's what we see down here so this is the basic form of an
lstm uh it has a bunch of formulas but essentially how you can picture that is
is there is some sort of a hidden State and there is some sort of an input
and what I want to do is I will output a new hidden State and I need to somehow
add I need to somehow somehow decide from this hidden state that comes in
what do I forget so I need to decide strategically part of it I will forget
and then of the input here I strategic ially decide what part I want to add
let's say I want to only add the the bottom left here right and so that gives
me something like this and then I need to find a way to smartly put those two
together to make efficient use of the memory and that's all this stuff here
does a lot of this stuff um are called Gates and they call this the constant
error carousel so they have a simplified version of this right here this here is
the hidden state or part of the Hidden state if you if you will they split the hidden State into two different things
one is called C and one is called H I guess C is for Carousel and H is for
hidden state but they're they're both hidden State you can think of them as I don't know concatenate them then it's
one vector but you can also just think of the Hidden State as being a tupal in an lstm um the C here is the is the
interesting part in this diagram or in this formula you can see that we take
the C from the last step and we multiply it by this F thing now this F thing is
just a number between zero and one by the way in the original formulation of the lstm all of this were just single
scalers so we just we're just wondering what scaler should we remember later it
was extended to vectors um in that case this would be a
piecewise uh piecewise multiplication right here so f is just a number between
zero and one and we call it the forget gate so if f is zero it means I forget C
if f is one it means I remember and likewise in the vector formulation if f
is zero in a particular Dimension I mean it means I forget that Dimension if f is one I remember it um if f is 0.5 it
means I forget it somewhat like I I weaken my memory of it right likewise
the I is called the input gate how much of the current input Z is the current
input how much of the current input do I want to remember again if I is zero I
don't remember it at all um if I is one I do I do remember the current input you
can see we add those together so F and I dictate how much do I want to forget of
the past and I regulates how much of the current thing I'm looking at do I want
to incorporate into my memory no these could both be one right there there's no
no necessary that they add to one or something like this um but these are
called Gates and these Gates made it possible to extend lstms without the
problem of what we would usually call exploding gradients the very basic lstms
were implemented like something like this like the the hidden State at time step t+1 was sort of like the um
was um you can you can
uh you you can choose you can uh say okay it was like the it was some some
sort of multiplication of the Hidden state of t plus so the W1 plus
W2 the current the current input right so you might just say well I know neural
networks so I know mult and I know like I just multiply by a weight and then I
may may even push it through some sort of nonlinearity right but if we leave that away and just look at the basics
just want to incorporate the last state and then I want to incorporate the current signal the problem is if you
look at this you're and you you do this over and over and over right you do this
t+ 1 for t + 2 you do the same again but now your H is already the result of a
previous multiplication so as you go through time you kind of multiply the same Matrix over and over and over and
over again which uh leads to if if the spectral values of this
Matrix are larger than one you just get an explosion because you essentially do the same multiplication with a number
greater than one over and over and over and over again if they are smaller than
one very quickly you go to zero so people had real problem getting
gradients to flow through these time steps because they would apply the same
multiplication over and over uh these gating functions like additive additive
updating of the Hidden State using gating functions on on the signals
meting miate that to a larger degree especially the sorry the exploding
gradients is something that is not necessarily a problem here cuz you're
adding you're not you're know constantly um multiplying right and you're only
gating you're not like here you don't multiply multiply multiply um all the
time all right then there are some nonlinearities in here so if you look at
the lstm formulation you can see these Gates they are computed from the current input they
are also computed from the last hidden state right so you take the gates are informed the gates are not just fixed
the gates are informed like how much do I want to forget is informed by the current data and is informed by the last
hidden State um even the even the current input is modulated by the
current data obviously like that that's the main input here but also by the last hidden State and so on so um and then
the you can see these gates are computed there is a sigmoid nonlinearity involved
which makes them between zero and one uh this is tanh I believe so there's a
couple of non or this here is tan H maybe yeah it could be uh and this here
is just I don't know here here they
are on they are the cell input activation functions typically tan H
okay and then you can see here this error Carousel as we just looked at it
and the hidden state is just produced we call it this is the output gate right
the hidden state is also the output of the current time step so every time step can give you an output value um you can
use that for next token prediction um and that's achieved by shoving the C
through a non linearity so here you see that the
dependence from C to the last C is de
facto linear right it's um you know if you think of how the
gradient flows there's no nonlinearity in the way it's linear you can just zoom
okay um and that for example is used in state Spas models the fact that current
time step to the next time step is linear like and how you aggregate data
is fixed and is not data dependent um that fact is used to be able to scale it
really efficiently however you look at the hidden state that is also passed
from time step to time step it's not the case so the hidden State um sorry the
hidden state is has nonlinearity on it is passed from time step to time step
and influences how the data is aggregated Into The Hidden state or or or even the the carousel so
this dependence of hidden state of the current step on the hidden state of the
past step that is a recurrent connection that has a nonlinearity inside of it and
that makes the whole thing not parallelizable I need to wait for the
last time step in the current layer before I can compute the next time step
in the current layer so if you compare this to Transformers up here if if
you're during training obviously if you're during inference you're doing next token prediction yes you need to wait for the you need to wait for the
previous thing to finish until you before you can start the next thing but during training where you have all the
tokens available you are able to just do all of this at the same time you just
for all the tokens not just for this one but for all the tokens at the same time
and then you're able to do the next layer all at the same time right and
that we call parallelizable training it's among the things that make Transformers really really really
powerful because you can see because you we have the time step
dependence between here we just have the time step dependence between the last
layer's last step and the next layer's current step in an lstm we also have
connections like this and this right here and these are these are the ones
that then um especially if they're nonlinear these are the ones that make
it really not train in a parallelizable fashion so if I have 100 tokens in a
sequence training an lstm on it is way slower than training a parallelizable
Transformer like with a Transformer I get 100 samples at the same time and in lstm I do time step by time step and I
only get one sample at a time all right okay let's go to the two
extensions that they do the SL lstm and the M lstm now the SL lstm you can see
right here the first thing they say they do is uh we broadcast the original lsdm
gating techniques um we yeah we broadcast the gating
techniques and if you look [Music]
at I think they say here okay um they're now going to do this in and that's not
that's not written here but they're now going to do this in a vector fashion so
these Gates here sorry are going to be vectors the Z signal is going to be a
vector and therefore they're now going to to use matrices here like w w w w
right uh big matrices that gives you a vector so the gates are vector then we
have elementwise multiplication by the gates instead of scalar multiplication and so on this unless I'm super mistaken
has already been done for many like 20 years or something like
this like it's very natural to do so to say the hidden state is actually a vector and the gates are vectors and
however doing this allows them to call this memory mixing so you see if you just apply this to scalers and you then
do it twice let's say you do you have two C's and two hes right so you have two scalers and you do all of this
individually it kind of means the current H only gets input from the last
ede of itself right like think I just have two of them but the hidden
connection the recurrent connection are only between themselves right I only pass to my corresponding next time step
H and then if you H over there right you only pass yours however if we have
matrices right here what matrices do unless they are strictly diagonal they
take information from some dimensions and they can apply it to other
dimensions right every off diagonal entry is essentially a information route between Dimensions so if I now consider
all of these to be vectors then that means the some like the dimension one of
the Hidden State can influence what's on Dimension four of the forget gate on the next state this is it's very natural
properties of matrices and you probably didn't even explicitly think of this but by them starting at the scaler variant
of lstms um and now they're getting to call
this memory mixing so they say oh we have this this new memory mixing thing
and um they can have multiple memory cells enable memory mixing via recurrent
connections from hidden State um yada y y yeah okay so they call that memory
mixing like the thing that people are already doing and then they go a step further
and say actually instead of making this a full Matrix we can make this a block diagonal matrix and then we have multi
head memory mixing because now we only have memory mixing inside of these heads
now that may very well be advantageous right instead of fully memory mix everything just
restricted to different you know heads um might might be really cool
but you know I I feel they they make the step here seem bigger than it is by
touting this as like new memory mixing architecture all right the second thing
they do is they replace the sigmo nonlinearity which was here and here
you'll remember by an exponential function and that that is actually
something new right the sigma nonlinearity is is good because if you
don't want exploding exploding gradients then the sigma nonlinearity is
pretty guaranteed to keep it between zero and one however it also is not so
good because it saturates very quickly like as soon as you go over here with your values your essentially a flat
function your gradient is essentially zero so it's good against gradient explosion but not so good against
Vanishing gradients and therefore they say hey what if we just take another nonlinearity that doesn't doesn't really
saturate uh and the exponential function is a good nonlinearity for that
now obviously the exponential function has a few problems namely it grows very
quickly and that's why they also so you can see we have this new line right here
this new n Line This is a normalizer and you can see that the the H H is now
oopsy H is now no longer computed as a nonlinear function of C but H is now
computed by dividing C by n and you can see what this n does is it simply sort
of it simply accumulates F and I because F and I now
can be potentially quite big numbers right keeping track of what they were in
the past gives so you you sort of pull through you aggregate the the signal but the
signal is multiplied by large constants so you also aggregate the constants
themselves um by constants here I don't mean constant constant but you know like multipliers you also aggregate them and
then each time step you simply divide one by the other in order to get a normalized output oh as long as you can
accumulate these things um you can sort of bring back from these giant numbers you can kind of bring it
back to a reasonable range by tracking that normalization state so that's is
pretty cool and it's an effective way of dealing with the exponential nonlinearity and you'll notice
conspicuously we've also yeah as I said dropped the nonlinearity here they could
still have one they could still have a nonlinearity right there they choose not to so making the whole thing more linear
um and also here this probably will help preventing
the kind of Vanishing gradient thing and since we already normalize and they also do some gradient clipping I recall uh
then we're probably fine lastly they introduce they actually
say actually we don't use the F and I you see above we're actually using the F and I that you see right here by
dividing you know this we're we're in kind of of log space right here and and subtract so we're essentially dividing
by this m term the M term is like the max of this and this and you can
calculate this out um it actually doesn't change anything about the output so the H value the output value of each
layer and the gradients uh do not change it's just kind of a a numerical trick
that they apply right here so lots of numerics and tricks in order to um
mitigate the fact that they've just put in an exponential function instead of a sigmoid okay so in summary the slsm cell
was actually new there is a new nonlinearity right here then there is
stuff you have to do to handle the fact that you just put in a new
nonlinearity they drop another nonlinearity and
they extend the memory mixing even though I I think that's a bit of an
overstatement so I'd rather say they introduce block diagonal matrices
instead of full matrices in their recurrent computations so by the way
also here r r r r w
yeah good that's one block the other block mlsm mlsm relies on these uh
associative memories so the question is I
have at my disposal a matrix d by D okay
I need to store vectors of dimensionality D inside of
the Matrix I I can um so I want to store them and I want to retrieve them again
so each Vector also comes with an equivalent key Vector so I want to store
the vector somehow like V and then I want to use the key in order to retrieve
it again now one thing you could do is I can just put the first Vector here and
remember that particular key is at column one I can put the second one here
remember that that particular key is a column two and so on so you can see that works perfectly until capacity is full
and I am at you know then no longer and then I'm back into the exact same
problem as with the lstm I'm like well which one do I need to override and buy how much right so is there a better way
there is or at least there there could be um so what we can do actually instead
of just randomly using the value and key we can actually build an outer product
of the value and the key so that gives us a matrix that Matrix is rank one but
it kind of represents the value and the key and
then we can add that to the memory we already have so the memory at the beginning is empty but then gives us a
matrix then we can have another value V2 with key2 we can make the outer product of
those two and add those
together oopsie that's not a matrix matrices aren't circles
and we can do that infinitely long now yes the information will kind of override each other because we stacked
them on top but what's actually happening here so I have a memory memory zero and I add to it
V1 K1 outer product okay that's my
memory one now let's say this is empty at the beginning so this is zero what happens
how do I retrieve now well I retrieve by multiplying my memory with the
key K1 okay what does that if I just multiply this out this is V1 K1
transposed K1 now what we can do is we can when we store stuff just make the
keys normalized like we just make them have length one in this case this here just reduces to one right this is a DOT
product so just reduces to the scaler one and I'm left with B1 perfect recall
let's say I take this M1 and I add another pair plus V2
K2 I get M2 now let's say again so M2 is
now has this and this inside of it let's say again I want to retrieve key1 so I
multiply M2 by key1 that gives me V1 K1 T
K1 plus V2 K2 transpose
K1 right cuz this is this and these here
are the ones I stored so what does that do well this again goes to one and this
here is the question so there we make use of the fact that in high
Dimensions if I have randomly distributed vectors which
we can make the keys be maybe sort of let's just assume they are they are
going to be approximately orthogonal that's just the property of high Dimensions is just random vectors they're
orthogonal to each other like unless they're really close they don't have they don't have a they don't have an
inner product so they're orthogonal so they go to zero so this is is so this
this Falls away this is is V1 is so we were able to go with key1
and again retrieve approximately V1 now the question is obviously yeah
with the way up here we were also like with stacking we were also able to store two vectors right but if you read into
the literature around this kind of memory under certain conditions and so on you can see that this is a more
optimal way of storing stuff like the the Distortion you get the approximations you get of what you get
out are quite good even after you fill in more vectors than you technically
have space for right you just keep adding rank one matrices on top of one another and um the memory capacity of
that is quite large in comparison to
other strategies so this is essentially just a way to store stuff in if you have
a given amount of bits in memory okay uh and
yeah um arbitrarily much stuff in that so this is what they do so instead of
instead of um updating a single Vector with you know hidden state with a vector
on top they are now they have a matrix memory so the c is now capital and they
update with this outer product um obviously if you go to the actual
implementation that's a bit more um Extended so what we do is we no longer just build
this Z value which represents kind of the current state but we now build the key and value this is analogous like the
terminology here is analogous to Transformer but they're just things that are computed from the current inputs and
that we then store in the memory at the same time we also produce queries from
the current memory and then those queries get multiplied to the current
memory right so this is now the retrieval step so we've just added the
current time step to the memory and now we go and we retrieve from the memory so
we retrieve from the current time step but also from the past um so we
retrieve um yeah from from this memory what interests us and that's what the queries
are for notice that this is all all linear uh so this
can be highly parallelized there is no nonlinear dependence on past steps and
so on um again we pull in this normalizer because we have exponential nonlinearities and you'll see here
rather than above above we had dependence on H from the last step we no
longer have that they're just like ah these recurrent connections on the nonlinear hidden State let's just not do
that how bad can it be so the only connection that goes from time step to
time step is this C and obviously the the normalizer they're all like additive
and have no nonlinearities on on this on the path here there's a nonlinearity in
f um but not not on the and the F only
depends on the current time step so this is notable right there is no Pas time that comes in here and
sneaks through this nonlinearity and gets here and and gets into the next hidden state right there's no
nonlinearity on the path from one hidden state to the next hidden State and that
allows you to do this parallelizable training because linear stuff is fast
stuff and uh is parallelizable stuff so yeah um again new
nonlinearity uh Matrix memory instead of single Vector memory then they
reformulate stuff a bit so it looks more like Transformers even though you could probably formulate it in a way that that
looks a lot more similar to the to the stuff up here um and uh and yeah and that's that's
that I wonder I wonder if they just increase the
dimensionality obviously now instead of D dimensional memory we have D byd dimensional memory I wonder what happens
if they just increase the hidden State like by you know Square the dimension on
that and then like up project and do just do lstm because you know like ultimately
you're just increasing the memory capacity here and yes you do it in this funky way which is really cool and you
you know like I'm on board with that but I wonder what happens if you just do
lstm you don't do hidden to Hidden like hidden nonlinear connection and you just
Square the dimensions on the hidden
state but notably um there is no parameters involved in this so this
pretty cool so there's parameters here in building like queries keys and values there's no parameters necessarily in the
kind of storing and receival retrieval um procedure as such so I guess if you
had to up project to huge Dimension you would also have to have huge amounts of of parameters so maybe I don't know I
don't know maybe that's the answer all right um then they wonder how do we make
this into blocks so now we we have no sorry now we have blocks right these things are blocks uh now how do we make
them into no yeah blocks yeah how do we make
blocks from so here we have have like an S lstm and here we have like an M uh
lstm and we wonder how how do we embed them into because Transformers they
don't just have attention they have like attention and residual connections and normalization and up projection right uh
the the feet forward layers and so on okay their answer is going to be for the slsm they found it to be beneficial to
just do that slsm stuff here and then do the this is this up here is is essentially a a what would be the feed
forward uh Network in a Transformer um and then here um sorry in
the M lstm uh they do they do the up projection of the feet forward Network
here then they do all of that stuff in the higher Dimension and they then they do the down projection after that so
they kind of up project and and then after that down project uh
so yeah they must have found that this is more more um advantageous if you
actually look at it like we can quickly look at that they have in the appendix they have the these actual drawings
[Music] uh all right the appendix is fairly extensive which is is good okay so you
see here uh there is more there's more to it so um you see there is incoming
signal there's layer norm and then the input and forget gate
computation for some reason just go to like a con of window size 4
convolutional layer followed by like an on linearity and then go into this block
diagonal um block diagonal multi head uh memory mixing thing and you can see this
here symbolizes recurrent connection um whereas the Z and O like the output gate
and the representation of the current input they don't go through that okay then we have a group norm and we have an
up projection and then there is like a um another nonlinearity in here and then
you have a down projection again so again this is a bit akin to the feed forward Network in a Transformer block
and you have a residual connection all around that whereas the M lstm Block you
see I first the up projection then again part of your like the queries and keys
here go through this um con and switch this might be to
substitute the missing softmax right because the softmax would exactly act on
the queries and and keys but yeah the values don't go through that then there there's a layer skip this is the another
residual connection that's just introduced on top of this residual connection right here um and
then I guess this here is is then akin to the feet forward layer of the
Transformer they more talk about preup project and postup project in this stuff
I guess they just tried around a little bit and they found that this works better okay obviously if you up project
first then your Matrix memory in here needs to be even bigger and that's what they tout as one of one of the main
limitations right here is that yes we don't do the Transformer memory thing we have constant memory however that
constant memory can be large so they have a linear computation and constant memory complexity with respect to the
sequence length um since it's compressive it's well suited for yada yada
um memory of the mlsm does not require parameters but is computationally expensive through its d byd m Matrix
memory and D byd update 


hello this is Yanik from the future unfortunately my
screen recording um OBS has crashed well
it was my mistake because my laptop ran out of battery and then I was interrupted and then uh when I came back
everything was crashed so I lost the second half of this video um but it was actually to the good because I had a
conversation had the opportunity to have a conversation with the authors uh since
so that was really cool um the next part that would come here are experimental
evaluations now the experimental evaluations are really thorough uh they
do post a lot of numbers on a lot of different tasks and so on um and what it
looks like is that it's not like a super clear winner but that's not what they
aim for but rather it is up there as you would say it is for some tasks
especially the tasks where recurrent neural networks like recurrency helps uh
it's really good at obviously and for sort of General language modeling and general llm evaluations you can say it
is um competitive with similar models now whether it's better worse a bit
better and whatnot I think that is left up to the exact evaluation and exact
Benchmark there has been um a bit of criticism in how they implemented the
baselines and compareed to the baselines for example it's the process is obviously
not very intro up to introspection it seems like uh there might have been some
you know not as much effort into the baselines as there was into their models which it's still a research paper and
yes that's how everybody does things however through the conversations with the authors um which again very lucky
that I was able to have that they've told me that they would update the paper soon um with updated numbers on the
baselines as far as I understand they have trained some of the baselines only
in 16bit Precision uh where it would be important to use mix precision and
that's what they manag to do now and so they they will update the paper with that so I'm going to leave out any sort
of comments on the evaluations as such just the global picture is that it is in
there in the current models like what you can say is that if you compare xlsm
to a to llama Mamba what whatnot models
then at least they're going to perform
similarly and yeah time will tell if people actually adopt this time will
tell if this is actually useful in real world setting time will tell if what
they say right here oh it's you can train it really well whether that's really the case I think that just
requires a lot of different people to either incorporate this or incorporate
parts of it into their deployments they do have a section on limit ations which is also very cool so and I just jumped
away from that and they also have a a section on conclusion so in the
limitations they do mention hey look um we have we we cannot do a fast parallel
implementation of training that's kind of out of the out of the scope I
cannot can I do this no I can't oh well um they do not they cannot do fast
parallel training because there is a recurrency involved uh nevertheless they
develop the fast Cuda kernel for that also if depending on how big you make
your M lstm memory obviously that alone is a lot of numbers
that require some crunching even if it's not recurrent uh number crunching is expensive and therefore the size the
extra size of that memory introduces this extra constant cost if you will constant Factor cost uh and then yeah
lastly they say due to the expensive computational load for large language experiments we didn't neither fully
optimize the architecture nor the hyper parameters especially for a larger xlsm architecture in the conclusion they say
their original question being how far do we get in language modeling when scaling lstm do billions of parameters so far we
can answer at least at as far as current Technologies like Transformers or States based models again whether whether
that's you know 100% accurate I think time will tell um what we can say at
least from this paper is it's it's in there somewhere they're also say they
would release code and maybe also models so people might be able to check for
themselves once that's out um yeah so this was it uh sorry that
the second second half here uh went went bust but I hope you were able to make
sense out of a bit of this and I wish you wonderful rest of the day by the way
if you interested in paper discussions we do paper discussions almost every Saturday on Discord in fact on our
Discord there are people doing paper discussions almost every day uh self organized very very cool very very big
effort of the people involved and yeah if you're interested come join and that's it bye-bye

Part 4
hello there today we're going to look at xlsm extended long short-term memory by
maximilan Beck corbinian Pepple and a team around them especially the
corresponding author here is Zep hiter who you may remember was the original or
among the original inventors of lstm's long shortterm memory neuron Network
the most popular type of current neural network there was invented in the '90s
obviously where all the good things were invented and this paper is a new take on
an lstm with all the learnings that we now have from the llm world like from
the world of Transformers and so on so the central question here is how how far
can we push uh lstm type architectures or recurrent architectures in general
and how much can they compete with Transformer based models or attention
based models when it comes to specifically language modeling so
obviously I I don't know but if like let's say if let's say there is something to this right then that would
be a huzzah back for the lstm authors uh
so there's also maybe a bit of of personal interest right here but I do
think it's a cool work in a cool Direction and notably it ties into a bigger picture uh around does it even
matter what architecture we have do only parameters matter you know what's really important in these kind of things and we
can get all into all of that a bit later but first note the beauty here the paper
starts with in the 1990s you know you know you're a you're about to read
history when a paper starts but with the word in the 1990s yeah in the 1990s I uh can't
really remember much of that time but some people can and this paper says the
constant error Carousel and gating were introduced as the central ideas of the
long shortterm memory since then lstms have stood the test of time contributed
to uh numerous deep learning success stories in particular they constituted
the first first large language models already were out of the gates with
extraordinary claims there is a I don't know there's something to be said about
lstms they certainly had their impact they had their impact in numerous domains right they they were widespread
their properties make them really good sequence processors even sequences that Transformers struggle with like time
series and I don't know like heart rate data and whatnot audio data all of this
stuff lstms could you know are applied they don't use uh humongous amount of
resources and so on and yes they were also used for language modeling notably
a lot of the um machine translation architectures and so on were powered by
lstms um yeah but I don't know the the
the the 1990s crew must kind of formulate everything that's happening
today as oh this was already done right so now the hype is around large language
models by which we mean specifically language models that take you know they
go really like big in parameters it really achieve their achieve their width
and achieve their performance by scaling right and then pumping as much data as
possible into them and yes you may say that some of the language models back
then and some of the translation models were kind of that but always being like
Oh we we did that we already did that back then we did Transformers back then we did Gans back then I don't want to
get into a fight here okay uh no opinions on those things I'm just saying
whether it's true or not the constant not just referral to hey that was great
back in the day but on top of that being like you know all of this stuff today we we actually did that already back in the
day yeah sorry rant over I do have a bit of an issue with the first large
language model I would I would strongly argue that the first large language
models were very probably the engram models that in particular Google built
on really large scales like the number if you count these as parameters the number of parameters of these were like
through the roof and therefore I would seriously consider the lsms to not
be the first large language models like as long as we're you know measuring
gonads um I would I don't know I'd be interested in in revising that however
the Advent of Transformer technology with parallelizable self attention at its core Mark the dawn of a new era
outpacing stms at scale we now raise a simple question how
far do we get in language modeling when scaling lstms to billions of parameters
and that's a valid question leveraging the latest techniques from Modern llms
but mitigating known limitations of lstms and that is is a really good
question right if we have all these learnings from Transformers some of them
aren't really particular to Transformers some of them are just like you need to normalize your
data in between layers in a smart way like yeah okay that that makes sense
right you need to use multiple heads if you do sort of single token processing
or if you you do processing it it pays off to have multiple heads in an operation that's fair we can do that
without Transformers right and a lot of architecture like State space models and
so on have actually implemented a lot of these things without Transformers and therefore you could say you know could
we apply some of that to lstms also could we kind of mitigate the
limitations that we know lstms have and if we do that what happens if we scale
them up are they going to be as good as Transformers are they going to be better are they going to be worse you know who
knows and the bigger question here is is that if you just have like some function
some trainable function um does it really matter how it's built or is it just the number of parameters that
matters given given that you train it like with enough data and you know well
with learning rates that fit and so on but does it matter or does it not matter
if it doesn't matter I mean lstms they do have constant memory usage and
therefore you know could be a great candidate they can go for infinitely long sequences could be a great
candidate all right they say we introduce exponential gating with appr appropriate normalization and
stabilization techniques secondly we modify the LSM memory structure obtaining two different things first SL
lstm with a scalar memory a scaler update and new memory mixing two M lstm
that is fully parallelizable with a matrix memory and the co-variance update rule y y then they do experiments the
experiments are good I can I can already tell you the their experiments are good and then the entire Community looks at
their experiments and goes like and we don't have code yet so
that's the current state where we're in uh so what they do is they take lstm
here on the left right and they modify it in two different ways like two separate ways of modifying the SL
lstm is um in the most plain sense it switches out some of the nonlinearities
some of the nonlinearities right now are sigmoids it switches them out via exponential function then it handles
some of the stuff that comes as downsides to using the exponential function because it it it grows pretty
quickly and then they also have this thing they call new memory mixing
however that just seems like it's uh it's a like that's just really new if you
look at the really really old old lstm but parts of that have kind of already
been been part of lstm world for many years um at least the way I understand
it secondly they changed the lstm to the mlsm note these are two separate things
right they derive two different cells um recurrent cells and then they make two
different blocks from from those and each block here then serves as a layer
in these stacked um in a in a stacked layer sense so here is depth of the
neural network and they can for example switch it with an M layer then an S layer then an M layer an M an S like
arbitrarily stack these layers on top of one another so whereas the Transformer
would only have like a single block that consists of a tension and and feed forward um this architecture has two
different blocks that it chains together in arbitrary fraction and that that's
how it propagates signal so the M lstm also has the exponential gating so again
switching out nonlinearities and then it has this Matrix memory The Matrix memory
is something we're going to look at notably if you know L lstms they have an internal state that they maintain so
they go over the sequence and they try to extract information from each each
element and put it into an internal State and then the later sequence elements can kind of read from that
internal state which is really different from the attention mechanism right um
what they do is here they extend they ask themselves how can we make this memory more have more
capacity um the naive way would just to be increase in dimensionality however
what they do right here is they have kind of an associative memory usage um
and we'll get to that and that comes with this covariance update rule uh and that then lends
itself to parallel training actually the parallel training is just a result that they kind of drop one connection from
like they drop one hidden to Hidden connection and then it's a it's a parallel and um doesn't seem to hurt so
they can they can just do it all right then they yeah they they G them notably
this m lstm thing is then parallelizable in training like a attention is so we have a lot to get into first maybe it's
um interesting to just see the difference between kind of what un like the the level that we care about what
attention mechanisms or Transformers do and then what kind of lstms do and why
why they have fallen out of fashion so in a Transformer and let's say we're
looking at a decoder only Transformer you have a sequence elements right and
then I I care about forward propagating signal across the sequence think of next
token prediction or something so I want to compute signal for this particular token here and what I do is in each
layer of my neural network I able to kind of look at all of the past using
attention right dynamically look at all of the past and aggregate information from that and then at the next layer
again I'm able to look at all of the past at the same time using the
attention mechanism and grab information dynamically right dynamically aggregate
information that I need now this has an advantage obviously I get to look at
every single piece of information every single time and
therefore that there's almost no better case for you know how can you work with
sequences every element gets to look at every other element in every single step
on the other hand you may also know that this does have considerable cost especially since this aggregation here
is uh consists of nonlinear functions um especially in attention I have I
multiply curries times keys I do the outer product and then I do a a softmax
out of that and then I multiply that by the values and this soft Max right here
is a highly nonlinear operation and therefore that kind
of that that denies me the opportunity to do a lot of optimizations so I have
to almost realize this Matrix in here which is really big or I have to do that
sequentially which costs time in turn an lstm would go over a
sequence one by one so let's say let's say I have I have a sequence um I want
to predict the next token the lstm would already have a hidden State somehow a
hidden State um I'm going to draw that here and
that hidden State already has information from all of the past that it
has accumulated one by one moving over the hidden state so the hidden State
already exists and all I need to do is I need to inte create information from the current element into the hidden
State and produce the next hidden State and from that I then also produce
produce the output of the current of the current step either from here or from here like from somewhere I produced the
output of the current step so each step only looks at itself and the last hidden
state so there's information that gets passed from time step to time step step and there is nothing in this that allows
me to explicitly look back at the past elements now this is good because it
means I just use constant memory right I have this hidden State and it's the same size as this hidden State and I just
need to find a way to update it so constant memory and I I need to learn a smart way to deal with that memory on
the other hand not being able to look at the individual sequence elements means that the what I encode from this state
into the next hit memory that's all there will ever be right like no future
thing can ever look at this element other than look at the information
that's available in the hidden state so it's almost like you need to learn to extract the exact right things that any
future any future element would want to know of you and you need to provide it
in a compact way and you so that it's addressable and you need to make sure
when you update the memory that you don't override anything that's important that's already there because you're
going to use the same bits to represent that you see the challenge that's why
originally people developed attention they said hey our rnns are good but they you know they would benefit from being
able to directly look back like at this element right here and that's how the attention mechanism was born and then
it's eventually people said actually we we don't need this this kind of hidden State thing anymore right now why was
this a benefit to not need the hidden State thing anymore and that's where we need to actually look at how the lstm is
built and that's what we see down here so this is the basic form of an
lstm uh it has a bunch of formulas but essentially how you can picture that is
is there is some sort of a hidden State and there is some sort of an input
and what I want to do is I will output a new hidden State and I need to somehow
add I need to somehow somehow decide from this hidden state that comes in
what do I forget so I need to decide strategically part of it I will forget
and then of the input here I strategic ially decide what part I want to add
let's say I want to only add the the bottom left here right and so that gives
me something like this and then I need to find a way to smartly put those two
together to make efficient use of the memory and that's all this stuff here
does a lot of this stuff um are called Gates and they call this the constant
error carousel so they have a simplified version of this right here this here is
the hidden state or part of the Hidden state if you if you will they split the hidden State into two different things
one is called C and one is called H I guess C is for Carousel and H is for
hidden state but they're they're both hidden State you can think of them as I don't know concatenate them then it's
one vector but you can also just think of the Hidden State as being a tupal in an lstm um the C here is the is the
interesting part in this diagram or in this formula you can see that we take
the C from the last step and we multiply it by this F thing now this F thing is
just a number between zero and one by the way in the original formulation of the lstm all of this were just single
scalers so we just we're just wondering what scaler should we remember later it
was extended to vectors um in that case this would be a
piecewise uh piecewise multiplication right here so f is just a number between
zero and one and we call it the forget gate so if f is zero it means I forget C
if f is one it means I remember and likewise in the vector formulation if f
is zero in a particular Dimension I mean it means I forget that Dimension if f is one I remember it um if f is 0.5 it
means I forget it somewhat like I I weaken my memory of it right likewise
the I is called the input gate how much of the current input Z is the current
input how much of the current input do I want to remember again if I is zero I
don't remember it at all um if I is one I do I do remember the current input you
can see we add those together so F and I dictate how much do I want to forget of
the past and I regulates how much of the current thing I'm looking at do I want
to incorporate into my memory no these could both be one right there there's no
no necessary that they add to one or something like this um but these are
called Gates and these Gates made it possible to extend lstms without the
problem of what we would usually call exploding gradients the very basic lstms
were implemented like something like this like the the hidden State at time step t+1 was sort of like the um
was um you can you can
uh you you can choose you can uh say okay it was like the it was some some
sort of multiplication of the Hidden state of t plus so the W1 plus
W2 the current the current input right so you might just say well I know neural
networks so I know mult and I know like I just multiply by a weight and then I
may may even push it through some sort of nonlinearity right but if we leave that away and just look at the basics
just want to incorporate the last state and then I want to incorporate the current signal the problem is if you
look at this you're and you you do this over and over and over right you do this
t+ 1 for t + 2 you do the same again but now your H is already the result of a
previous multiplication so as you go through time you kind of multiply the same Matrix over and over and over and
over again which uh leads to if if the spectral values of this
Matrix are larger than one you just get an explosion because you essentially do the same multiplication with a number
greater than one over and over and over and over again if they are smaller than
one very quickly you go to zero so people had real problem getting
gradients to flow through these time steps because they would apply the same
multiplication over and over uh these gating functions like additive additive
updating of the Hidden State using gating functions on on the signals
meting miate that to a larger degree especially the sorry the exploding
gradients is something that is not necessarily a problem here cuz you're
adding you're not you're know constantly um multiplying right and you're only
gating you're not like here you don't multiply multiply multiply um all the
time all right then there are some nonlinearities in here so if you look at
the lstm formulation you can see these Gates they are computed from the current input they
are also computed from the last hidden state right so you take the gates are informed the gates are not just fixed
the gates are informed like how much do I want to forget is informed by the current data and is informed by the last
hidden State um even the even the current input is modulated by the
current data obviously like that that's the main input here but also by the last hidden State and so on so um and then
the you can see these gates are computed there is a sigmoid nonlinearity involved
which makes them between zero and one uh this is tanh I believe so there's a
couple of non or this here is tan H maybe yeah it could be uh and this here
is just I don't know here here they
are on they are the cell input activation functions typically tan H
okay and then you can see here this error Carousel as we just looked at it
and the hidden state is just produced we call it this is the output gate right
the hidden state is also the output of the current time step so every time step can give you an output value um you can
use that for next token prediction um and that's achieved by shoving the C
through a non linearity so here you see that the
dependence from C to the last C is de
facto linear right it's um you know if you think of how the
gradient flows there's no nonlinearity in the way it's linear you can just zoom
okay um and that for example is used in state Spas models the fact that current
time step to the next time step is linear like and how you aggregate data
is fixed and is not data dependent um that fact is used to be able to scale it
really efficiently however you look at the hidden state that is also passed
from time step to time step it's not the case so the hidden State um sorry the
hidden state is has nonlinearity on it is passed from time step to time step
and influences how the data is aggregated Into The Hidden state or or or even the the carousel so
this dependence of hidden state of the current step on the hidden state of the
past step that is a recurrent connection that has a nonlinearity inside of it and
that makes the whole thing not parallelizable I need to wait for the
last time step in the current layer before I can compute the next time step
in the current layer so if you compare this to Transformers up here if if
you're during training obviously if you're during inference you're doing next token prediction yes you need to wait for the you need to wait for the
previous thing to finish until you before you can start the next thing but during training where you have all the
tokens available you are able to just do all of this at the same time you just
for all the tokens not just for this one but for all the tokens at the same time
and then you're able to do the next layer all at the same time right and
that we call parallelizable training it's among the things that make Transformers really really really
powerful because you can see because you we have the time step
dependence between here we just have the time step dependence between the last
layer's last step and the next layer's current step in an lstm we also have
connections like this and this right here and these are these are the ones
that then um especially if they're nonlinear these are the ones that make
it really not train in a parallelizable fashion so if I have 100 tokens in a
sequence training an lstm on it is way slower than training a parallelizable
Transformer like with a Transformer I get 100 samples at the same time and in lstm I do time step by time step and I
only get one sample at a time all right okay let's go to the two
extensions that they do the SL lstm and the M lstm now the SL lstm you can see
right here the first thing they say they do is uh we broadcast the original lsdm
gating techniques um we yeah we broadcast the gating
techniques and if you look [Music]
at I think they say here okay um they're now going to do this in and that's not
that's not written here but they're now going to do this in a vector fashion so
these Gates here sorry are going to be vectors the Z signal is going to be a
vector and therefore they're now going to to use matrices here like w w w w
right uh big matrices that gives you a vector so the gates are vector then we
have elementwise multiplication by the gates instead of scalar multiplication and so on this unless I'm super mistaken
has already been done for many like 20 years or something like
this like it's very natural to do so to say the hidden state is actually a vector and the gates are vectors and
however doing this allows them to call this memory mixing so you see if you just apply this to scalers and you then
do it twice let's say you do you have two C's and two hes right so you have two scalers and you do all of this
individually it kind of means the current H only gets input from the last
ede of itself right like think I just have two of them but the hidden
connection the recurrent connection are only between themselves right I only pass to my corresponding next time step
H and then if you H over there right you only pass yours however if we have
matrices right here what matrices do unless they are strictly diagonal they
take information from some dimensions and they can apply it to other
dimensions right every off diagonal entry is essentially a information route between Dimensions so if I now consider
all of these to be vectors then that means the some like the dimension one of
the Hidden State can influence what's on Dimension four of the forget gate on the next state this is it's very natural
properties of matrices and you probably didn't even explicitly think of this but by them starting at the scaler variant
of lstms um and now they're getting to call
this memory mixing so they say oh we have this this new memory mixing thing
and um they can have multiple memory cells enable memory mixing via recurrent
connections from hidden State um yada y y yeah okay so they call that memory
mixing like the thing that people are already doing and then they go a step further
and say actually instead of making this a full Matrix we can make this a block diagonal matrix and then we have multi
head memory mixing because now we only have memory mixing inside of these heads
now that may very well be advantageous right instead of fully memory mix everything just
restricted to different you know heads um might might be really cool
but you know I I feel they they make the step here seem bigger than it is by
touting this as like new memory mixing architecture all right the second thing
they do is they replace the sigmo nonlinearity which was here and here
you'll remember by an exponential function and that that is actually
something new right the sigma nonlinearity is is good because if you
don't want exploding exploding gradients then the sigma nonlinearity is
pretty guaranteed to keep it between zero and one however it also is not so
good because it saturates very quickly like as soon as you go over here with your values your essentially a flat
function your gradient is essentially zero so it's good against gradient explosion but not so good against
Vanishing gradients and therefore they say hey what if we just take another nonlinearity that doesn't doesn't really
saturate uh and the exponential function is a good nonlinearity for that
now obviously the exponential function has a few problems namely it grows very
quickly and that's why they also so you can see we have this new line right here
this new n Line This is a normalizer and you can see that the the H H is now
oopsy H is now no longer computed as a nonlinear function of C but H is now
computed by dividing C by n and you can see what this n does is it simply sort
of it simply accumulates F and I because F and I now
can be potentially quite big numbers right keeping track of what they were in
the past gives so you you sort of pull through you aggregate the the signal but the
signal is multiplied by large constants so you also aggregate the constants
themselves um by constants here I don't mean constant constant but you know like multipliers you also aggregate them and
then each time step you simply divide one by the other in order to get a normalized output oh as long as you can
accumulate these things um you can sort of bring back from these giant numbers you can kind of bring it
back to a reasonable range by tracking that normalization state so that's is
pretty cool and it's an effective way of dealing with the exponential nonlinearity and you'll notice
conspicuously we've also yeah as I said dropped the nonlinearity here they could
still have one they could still have a nonlinearity right there they choose not to so making the whole thing more linear
um and also here this probably will help preventing
the kind of Vanishing gradient thing and since we already normalize and they also do some gradient clipping I recall uh
then we're probably fine lastly they introduce they actually
say actually we don't use the F and I you see above we're actually using the F and I that you see right here by
dividing you know this we're we're in kind of of log space right here and and subtract so we're essentially dividing
by this m term the M term is like the max of this and this and you can
calculate this out um it actually doesn't change anything about the output so the H value the output value of each
layer and the gradients uh do not change it's just kind of a a numerical trick
that they apply right here so lots of numerics and tricks in order to um
mitigate the fact that they've just put in an exponential function instead of a sigmoid okay so in summary the slsm cell
was actually new there is a new nonlinearity right here then there is
stuff you have to do to handle the fact that you just put in a new
nonlinearity they drop another nonlinearity and
they extend the memory mixing even though I I think that's a bit of an
overstatement so I'd rather say they introduce block diagonal matrices
instead of full matrices in their recurrent computations so by the way
also here r r r r w
yeah good that's one block the other block mlsm mlsm relies on these uh
associative memories so the question is I
have at my disposal a matrix d by D okay
I need to store vectors of dimensionality D inside of
the Matrix I I can um so I want to store them and I want to retrieve them again
so each Vector also comes with an equivalent key Vector so I want to store
the vector somehow like V and then I want to use the key in order to retrieve
it again now one thing you could do is I can just put the first Vector here and
remember that particular key is at column one I can put the second one here
remember that that particular key is a column two and so on so you can see that works perfectly until capacity is full
and I am at you know then no longer and then I'm back into the exact same
problem as with the lstm I'm like well which one do I need to override and buy how much right so is there a better way
there is or at least there there could be um so what we can do actually instead
of just randomly using the value and key we can actually build an outer product
of the value and the key so that gives us a matrix that Matrix is rank one but
it kind of represents the value and the key and
then we can add that to the memory we already have so the memory at the beginning is empty but then gives us a
matrix then we can have another value V2 with key2 we can make the outer product of
those two and add those
together oopsie that's not a matrix matrices aren't circles
and we can do that infinitely long now yes the information will kind of override each other because we stacked
them on top but what's actually happening here so I have a memory memory zero and I add to it
V1 K1 outer product okay that's my
memory one now let's say this is empty at the beginning so this is zero what happens
how do I retrieve now well I retrieve by multiplying my memory with the
key K1 okay what does that if I just multiply this out this is V1 K1
transposed K1 now what we can do is we can when we store stuff just make the
keys normalized like we just make them have length one in this case this here just reduces to one right this is a DOT
product so just reduces to the scaler one and I'm left with B1 perfect recall
let's say I take this M1 and I add another pair plus V2
K2 I get M2 now let's say again so M2 is
now has this and this inside of it let's say again I want to retrieve key1 so I
multiply M2 by key1 that gives me V1 K1 T
K1 plus V2 K2 transpose
K1 right cuz this is this and these here
are the ones I stored so what does that do well this again goes to one and this
here is the question so there we make use of the fact that in high
Dimensions if I have randomly distributed vectors which
we can make the keys be maybe sort of let's just assume they are they are
going to be approximately orthogonal that's just the property of high Dimensions is just random vectors they're
orthogonal to each other like unless they're really close they don't have they don't have a they don't have an
inner product so they're orthogonal so they go to zero so this is is so this
this Falls away this is is V1 is so we were able to go with key1
and again retrieve approximately V1 now the question is obviously yeah
with the way up here we were also like with stacking we were also able to store two vectors right but if you read into
the literature around this kind of memory under certain conditions and so on you can see that this is a more
optimal way of storing stuff like the the Distortion you get the approximations you get of what you get
out are quite good even after you fill in more vectors than you technically
have space for right you just keep adding rank one matrices on top of one another and um the memory capacity of
that is quite large in comparison to
other strategies so this is essentially just a way to store stuff in if you have
a given amount of bits in memory okay uh and
yeah um arbitrarily much stuff in that so this is what they do so instead of
instead of um updating a single Vector with you know hidden state with a vector
on top they are now they have a matrix memory so the c is now capital and they
update with this outer product um obviously if you go to the actual
implementation that's a bit more um Extended so what we do is we no longer just build
this Z value which represents kind of the current state but we now build the key and value this is analogous like the
terminology here is analogous to Transformer but they're just things that are computed from the current inputs and
that we then store in the memory at the same time we also produce queries from
the current memory and then those queries get multiplied to the current
memory right so this is now the retrieval step so we've just added the
current time step to the memory and now we go and we retrieve from the memory so
we retrieve from the current time step but also from the past um so we
retrieve um yeah from from this memory what interests us and that's what the queries
are for notice that this is all all linear uh so this
can be highly parallelized there is no nonlinear dependence on past steps and
so on um again we pull in this normalizer because we have exponential nonlinearities and you'll see here
rather than above above we had dependence on H from the last step we no
longer have that they're just like ah these recurrent connections on the nonlinear hidden State let's just not do
that how bad can it be so the only connection that goes from time step to
time step is this C and obviously the the normalizer they're all like additive
and have no nonlinearities on on this on the path here there's a nonlinearity in
f um but not not on the and the F only
depends on the current time step so this is notable right there is no Pas time that comes in here and
sneaks through this nonlinearity and gets here and and gets into the next hidden state right there's no
nonlinearity on the path from one hidden state to the next hidden State and that
allows you to do this parallelizable training because linear stuff is fast
stuff and uh is parallelizable stuff so yeah um again new
nonlinearity uh Matrix memory instead of single Vector memory then they
reformulate stuff a bit so it looks more like Transformers even though you could probably formulate it in a way that that
looks a lot more similar to the to the stuff up here um and uh and yeah and that's that's
that I wonder I wonder if they just increase the
dimensionality obviously now instead of D dimensional memory we have D byd dimensional memory I wonder what happens
if they just increase the hidden State like by you know Square the dimension on
that and then like up project and do just do lstm because you know like ultimately
you're just increasing the memory capacity here and yes you do it in this funky way which is really cool and you
you know like I'm on board with that but I wonder what happens if you just do
lstm you don't do hidden to Hidden like hidden nonlinear connection and you just
Square the dimensions on the hidden
state but notably um there is no parameters involved in this so this
pretty cool so there's parameters here in building like queries keys and values there's no parameters necessarily in the
kind of storing and receival retrieval um procedure as such so I guess if you
had to up project to huge Dimension you would also have to have huge amounts of of parameters so maybe I don't know I
don't know maybe that's the answer all right um then they wonder how do we make
this into blocks so now we we have no sorry now we have blocks right these things are blocks uh now how do we make
them into no yeah blocks yeah how do we make
blocks from so here we have have like an S lstm and here we have like an M uh
lstm and we wonder how how do we embed them into because Transformers they
don't just have attention they have like attention and residual connections and normalization and up projection right uh
the the feet forward layers and so on okay their answer is going to be for the slsm they found it to be beneficial to
just do that slsm stuff here and then do the this is this up here is is essentially a a what would be the feed
forward uh Network in a Transformer um and then here um sorry in
the M lstm uh they do they do the up projection of the feet forward Network
here then they do all of that stuff in the higher Dimension and they then they do the down projection after that so
they kind of up project and and then after that down project uh
so yeah they must have found that this is more more um advantageous if you
actually look at it like we can quickly look at that they have in the appendix they have the these actual drawings
[Music] uh all right the appendix is fairly extensive which is is good okay so you
see here uh there is more there's more to it so um you see there is incoming
signal there's layer norm and then the input and forget gate
computation for some reason just go to like a con of window size 4
convolutional layer followed by like an on linearity and then go into this block
diagonal um block diagonal multi head uh memory mixing thing and you can see this
here symbolizes recurrent connection um whereas the Z and O like the output gate
and the representation of the current input they don't go through that okay then we have a group norm and we have an
up projection and then there is like a um another nonlinearity in here and then
you have a down projection again so again this is a bit akin to the feed forward Network in a Transformer block
and you have a residual connection all around that whereas the M lstm Block you
see I first the up projection then again part of your like the queries and keys
here go through this um con and switch this might be to
substitute the missing softmax right because the softmax would exactly act on
the queries and and keys but yeah the values don't go through that then there there's a layer skip this is the another
residual connection that's just introduced on top of this residual connection right here um and
then I guess this here is is then akin to the feet forward layer of the
Transformer they more talk about preup project and postup project in this stuff
I guess they just tried around a little bit and they found that this works better okay obviously if you up project
first then your Matrix memory in here needs to be even bigger and that's what they tout as one of one of the main
limitations right here is that yes we don't do the Transformer memory thing we have constant memory however that
constant memory can be large so they have a linear computation and constant memory complexity with respect to the
sequence length um since it's compressive it's well suited for yada yada
um memory of the mlsm does not require parameters but is computationally expensive through its d byd m Matrix
memory and D byd update 

hello this is Yanik from the future unfortunately my
screen recording um OBS has crashed well
it was my mistake because my laptop ran out of battery and then I was interrupted and then uh when I came back
everything was crashed so I lost the second half of this video um but it was actually to the good because I had a
conversation had the opportunity to have a conversation with the authors uh since
so that was really cool um the next part that would come here are experimental
evaluations now the experimental evaluations are really thorough uh they
do post a lot of numbers on a lot of different tasks and so on um and what it
looks like is that it's not like a super clear winner but that's not what they
aim for but rather it is up there as you would say it is for some tasks
especially the tasks where recurrent neural networks like recurrency helps uh
it's really good at obviously and for sort of General language modeling and general llm evaluations you can say it
is um competitive with similar models now whether it's better worse a bit
better and whatnot I think that is left up to the exact evaluation and exact
Benchmark there has been um a bit of criticism in how they implemented the
baselines and compareed to the baselines for example it's the process is obviously
not very intro up to introspection it seems like uh there might have been some
you know not as much effort into the baselines as there was into their models which it's still a research paper and
yes that's how everybody does things however through the conversations with the authors um which again very lucky
that I was able to have that they've told me that they would update the paper soon um with updated numbers on the
baselines as far as I understand they have trained some of the baselines only
in 16bit Precision uh where it would be important to use mix precision and
that's what they manag to do now and so they they will update the paper with that so I'm going to leave out any sort
of comments on the evaluations as such just the global picture is that it is in
there in the current models like what you can say is that if you compare xlsm
to a to llama Mamba what whatnot models
then at least they're going to perform
similarly and yeah time will tell if people actually adopt this time will
tell if this is actually useful in real world setting time will tell if what
they say right here oh it's you can train it really well whether that's really the case I think that just
requires a lot of different people to either incorporate this or incorporate
parts of it into their deployments they do have a section on limit ations which is also very cool so and I just jumped
away from that and they also have a a section on conclusion so in the
limitations they do mention hey look um we have we we cannot do a fast parallel
implementation of training that's kind of out of the out of the scope I
cannot can I do this no I can't oh well um they do not they cannot do fast
parallel training because there is a recurrency involved uh nevertheless they
develop the fast Cuda kernel for that also if depending on how big you make
your M lstm memory obviously that alone is a lot of numbers
that require some crunching even if it's not recurrent uh number crunching is expensive and therefore the size the
extra size of that memory introduces this extra constant cost if you will constant Factor cost uh and then yeah
lastly they say due to the expensive computational load for large language experiments we didn't neither fully
optimize the architecture nor the hyper parameters especially for a larger xlsm architecture in the conclusion they say
their original question being how far do we get in language modeling when scaling lstm do billions of parameters so far we
can answer at least at as far as current Technologies like Transformers or States based models again whether whether
that's you know 100% accurate I think time will tell um what we can say at
least from this paper is it's it's in there somewhere they're also say they
would release code and maybe also models so people might be able to check for
themselves once that's out um yeah so this was it uh sorry that
the second second half here uh went went bust but I hope you were able to make
sense out of a bit of this and I wish you wonderful rest of the day by the way
if you interested in paper discussions we do paper discussions almost every Saturday on Discord in fact on our
Discord there are people doing paper discussions almost every day uh self organized very very cool very very big
effort of the people involved and yeah if you're interested come join and that's it bye-bye
when the well runs dry you might be
thirsty but this still StatQuest you
can watch it StatQuest 

Part 5:
hello I'm Josh
Starmar and welcome to StatQuest today
we're gonna continue our series on
machine learning fundamentals and we're
going to talk about sensitivity and
specificity they're gonna be clearly
explained this StatQuest follows up on
the one that describes the confusion
matrix so if you're not already down
with that
check out the quest the first half of
this video will explain how to calculate
and interpret sensitivity and
specificity when you have a confusion
matrix with two rows and two columns and
the second half will show you how to
calculate and interpret sensitivity and
specificity when you have three or more
rows and columns
even if you're already down with the
confusion matrix let's remember that
rows correspond to what was predicted
and columns correspond to the known
truth when there are only two categories
to choose from in this case the two
choices were has heart disease or does
not have heart disease then the top
left-hand corner contains the true
positives true positives are patients
that had heart disease that were also
predicted to have heart disease true
negatives are in the bottom right hand
corner true negatives are patients that
did not have heart disease and were
predicted not to have heart disease the
bottom left-hand corner contains the
false negatives false negatives are when
a patient has heart disease but the
prediction said they didn't lastly the
top right hand corner contains the false
positives false positives are patients
that do not have heart disease but the
prediction says that they do once we
filled out the confusion matrix we can
calculate two useful metrics sensitivity
and specificity in this case sensitivity
tells us what percentage of patients
with heart disease were correctly
identified
sensitivity is the true positives
divided by the sum of the true positives
and the false negatives
specificity tells us what percentage of
patients without heart disease were
correctly identified specificity are the
true negatives divided by the sum of the
true negatives and the false positives
in the StatQuest on the confusion
matrix we applied logistic regression to
a testing data set and ended up with
this confusion matrix let's start by
calculating sensitivity for this
logistic regression here's the formula
for sensitivity and for true positives
we plug in 139 and for false negatives
we plug in 32 when we do the math we get
zero point eight one sensitivity tells
us that 81% of the people with heart
disease were correctly identified by the
logistic regression model
now let's calculate the specificity
here's the formula for specificity and
for true negatives we will plug in 112 and
for false positives we will plug in 20 when
we do the math we get 0.85 specificity
tells us that 85% of the people without
heart disease were correctly identified
by the logistic regression model now
let's calculate sensitivity and
specificity for the random forest model
that we used in the confusion matrix
StatQuest
here's the confusion matrix here's the
formula for sensitivity and when we plug
in the numbers we get zero point eight
three
here's the formula for specificity and
when we plug in the numbers we get zero
point eight three again
now we can compare the sensitivity and
specificity values that we calculated
for the logistic regression to the
values we calculated for the random
forest
sensitivity tells us that the random
forest is slightly better at correctly
identifying positives which in this case
are patients with heart disease
specificity tells us that logistic
regression is slightly better correctly
identifying negatives which in this case
are patients without heart disease we
would choose the logistic regression
model
if correctly identifying patients
without heart disease was more important
than correctly identifying patients with
heart disease
alternatively we would choose the random
forest model if correctly identifying
patients with heart disease was more
important than correctly identifying
patients without heart disease BAM
in the confusion matrix stat quest we
calculated this confusion matrix when we
tried to predict someone's favorite movie
now let's talk about how to calculate
sensitivity and specificity when we have
a confusion matrix with three rows and
three columns the big difference when
calculating sensitivity and specificity
for larger confusion matrices is that
there are no single values that work for
the entire matrix instead we calculate a
different sensitivity and specificity
for each category
so for this confusion matrix we'll need
to calculate sensitivity and specificity
for the movie troll 2 for the movie Gore
police and for the movie cool as ice
let's start by calculating sensitivity
for troll 2 for troll 2 there were 12
true positives people that were
correctly predicted to love troll 2 more
than Gore police and cool as ice
so for true positives we'll plug in 12
and there were 112 plus 83 which equals
195 false negatives people that love to
troll 2 but were predicted to love
Gore police or cool as ice
so for false negatives will plug in 195
and when we do the math we get 0.06
sensitivity for troll 2 tells us that
only 6% of the people that loved the
movie troll 2 more than Gore police or
cool as ice were correctly identified
now let's calculate the specificity for
troll 2 there were 23 plus 77 plus 92
plus 17 equals 209 true negatives people
that were correctly predicted to like
Gore police or cool as ice more than
troll 2 so for true negatives will plug
in 209
and there were 102 plus 93 equals 195
false positives people that loved gore
police or cool as ice the most but were
predicted to love troll 2 so for false
positives will plug in 195 and when we
do the math we get 0.52 specificity
for troll 2 tells us that 52% of the
people who loved Gore police or cool as
ice more than troll 2 were correctly
identified
calculating sensitivity and specificity
for Gore police is very similar let's
start by calculating sensitivity
there are 23 true positives people that
were correctly predicted to love Gore
police the most
and 102 plus 92 equals 194 false
negatives people who loved Gore police
the most but were predicted to love
troll 2 or cool as ice more when we do the
math we get 0.11 sensitivity
for Gore police tells us that only 11%
of the people that loved Gore police
were correctly identified now let's
calculate specificity there were 12 plus
93 plus 83 plus 17 equals 205 true
negatives people correctly identified as
loving troll 2 or cool as ice more than
gore police
and 112 plus 77 equals 189 false
positives people predicted to love gore
police even though they didn't and when
we do the math we get 0.52 specificity
for gore police tells us that 52% of the
people that loved troll 2 or cool as
ice more than gore police were correctly
identified
lastly calculating sensitivity and
specificity for cool as ice follows the
same steps we identify the true
positives the false positives the true
negatives and the false negatives and
then plug in the numbers first for
sensitivity then for specificity double bam
if we had a confusion matrix with
four rows and four columns then we would
have to calculate sensitivity and
specificity for four different
categories little bam
in summary sensitivity equals the true positives
divided by the sum of the true positives
and the false negatives and specificity
equals the true negatives divided by the
sum of the true negatives and the false
positives we can use sensitivity and
specificity to help us decide which
machine learning method would be best
for our data if correctly identifying
positives is the most important thing to
do with the data we should choose a
method with higher sensitivity if
correctly identifying negatives is more
important then we should put more
emphasis on specificity
hooray we've made it to the end of
another exciting StatQuest if you like
this StatQuest and want to see more
please subscribe and if you want to
support stack quest well consider buying
one or two of my original songs alright
until next time quest on
when the well runs dry you might be
thirsty but this still StatQuest you
can watch it StatQuest 

hello I'm Josh
Starmar and welcome to StatQuest today
we're gonna continue our series on
machine learning fundamentals and we're
going to talk about sensitivity and
specificity they're gonna be clearly
explained this StatQuest follows up on
the one that describes the confusion
matrix so if you're not already down
with that
check out the quest the first half of
this video will explain how to calculate
and interpret sensitivity and
specificity when you have a confusion
matrix with two rows and two columns and
the second half will show you how to
calculate and interpret sensitivity and
specificity when you have three or more
rows and columns
even if you're already down with the
confusion matrix let's remember that
rows correspond to what was predicted
and columns correspond to the known
truth when there are only two categories
to choose from in this case the two
choices were has heart disease or does
not have heart disease then the top
left-hand corner contains the true
positives true positives are patients
that had heart disease that were also
predicted to have heart disease true
negatives are in the bottom right hand
corner true negatives are patients that
did not have heart disease and were
predicted not to have heart disease the
bottom left-hand corner contains the
false negatives false negatives are when
a patient has heart disease but the
prediction said they didn't lastly the
top right hand corner contains the false
positives false positives are patients
that do not have heart disease but the
prediction says that they do once we
filled out the confusion matrix we can
calculate two useful metrics sensitivity
and specificity in this case sensitivity
tells us what percentage of patients
with heart disease were correctly
identified
sensitivity is the true positives
divided by the sum of the true positives
and the false negatives
specificity tells us what percentage of
patients without heart disease were
correctly identified specificity are the
true negatives divided by the sum of the
true negatives and the false positives
in the StatQuest on the confusion
matrix we applied logistic regression to
a testing data set and ended up with
this confusion matrix let's start by
calculating sensitivity for this
logistic regression here's the formula
for sensitivity and for true positives
we plug in 139 and for false negatives
we plug in 32 when we do the math we get
zero point eight one sensitivity tells
us that 81% of the people with heart
disease were correctly identified by the
logistic regression model
now let's calculate the specificity
here's the formula for specificity and
for true negatives we will plug in 112 and
for false positives we will plug in 20 when
we do the math we get 0.85 specificity
tells us that 85% of the people without
heart disease were correctly identified
by the logistic regression model now
let's calculate sensitivity and
specificity for the random forest model
that we used in the confusion matrix
StatQuest
here's the confusion matrix here's the
formula for sensitivity and when we plug
in the numbers we get zero point eight
three
here's the formula for specificity and
when we plug in the numbers we get zero
point eight three again
now we can compare the sensitivity and
specificity values that we calculated
for the logistic regression to the
values we calculated for the random
forest
sensitivity tells us that the random
forest is slightly better at correctly
identifying positives which in this case
are patients with heart disease
specificity tells us that logistic
regression is slightly better correctly
identifying negatives which in this case
are patients without heart disease we
would choose the logistic regression
model
if correctly identifying patients
without heart disease was more important
than correctly identifying patients with
heart disease
alternatively we would choose the random
forest model if correctly identifying
patients with heart disease was more
important than correctly identifying
patients without heart disease BAM
in the confusion matrix stat quest we
calculated this confusion matrix when we
tried to predict someone's favorite movie
now let's talk about how to calculate
sensitivity and specificity when we have
a confusion matrix with three rows and
three columns the big difference when
calculating sensitivity and specificity
for larger confusion matrices is that
there are no single values that work for
the entire matrix instead we calculate a
different sensitivity and specificity
for each category
so for this confusion matrix we'll need
to calculate sensitivity and specificity
for the movie troll 2 for the movie Gore
police and for the movie cool as ice
let's start by calculating sensitivity
for troll 2 for troll 2 there were 12
true positives people that were
correctly predicted to love troll 2 more
than Gore police and cool as ice
so for true positives we'll plug in 12
and there were 112 plus 83 which equals
195 false negatives people that love to
troll 2 but were predicted to love
Gore police or cool as ice
so for false negatives will plug in 195
and when we do the math we get 0.06
sensitivity for troll 2 tells us that
only 6% of the people that loved the
movie troll 2 more than Gore police or
cool as ice were correctly identified
now let's calculate the specificity for
troll 2 there were 23 plus 77 plus 92
plus 17 equals 209 true negatives people
that were correctly predicted to like
Gore police or cool as ice more than
troll 2 so for true negatives will plug
in 209
and there were 102 plus 93 equals 195
false positives people that loved gore
police or cool as ice the most but were
predicted to love troll 2 so for false
positives will plug in 195 and when we
do the math we get 0.52 specificity
for troll 2 tells us that 52% of the
people who loved Gore police or cool as
ice more than troll 2 were correctly
identified
calculating sensitivity and specificity
for Gore police is very similar let's
start by calculating sensitivity
there are 23 true positives people that
were correctly predicted to love Gore
police the most
and 102 plus 92 equals 194 false
negatives people who loved Gore police
the most but were predicted to love
troll 2 or cool as ice more when we do the
math we get 0.11 sensitivity
for Gore police tells us that only 11%
of the people that loved Gore police
were correctly identified now let's
calculate specificity there were 12 plus
93 plus 83 plus 17 equals 205 true
negatives people correctly identified as
loving troll 2 or cool as ice more than
gore police
and 112 plus 77 equals 189 false
positives people predicted to love gore
police even though they didn't and when
we do the math we get 0.52 specificity
for gore police tells us that 52% of the
people that loved troll 2 or cool as
ice more than gore police were correctly
identified
lastly calculating sensitivity and
specificity for cool as ice follows the
same steps we identify the true
positives the false positives the true
negatives and the false negatives and
then plug in the numbers first for
sensitivity then for specificity double bam
if we had a confusion matrix with
four rows and four columns then we would
have to calculate sensitivity and
specificity for four different
categories little bam
in summary sensitivity equals the true positives
divided by the sum of the true positives
and the false negatives and specificity
equals the true negatives divided by the
sum of the true negatives and the false
positives we can use sensitivity and
specificity to help us decide which
machine learning method would be best
for our data if correctly identifying
positives is the most important thing to
do with the data we should choose a
method with higher sensitivity if
correctly identifying negatives is more
important then we should put more
emphasis on specificity
hooray we've made it to the end of
another exciting StatQuest if you like
this StatQuest and want to see more
please subscribe and if you want to
support stack quest well consider buying
one or two of my original songs alright
until next time quest on
vectors if you clicked on this video you
probably already know what a vector is
however regardless on your current level
of knowledge about them we were still
going to quickly reintroduce them in a
more genetic way so what is a vector
technically we could just give the
definition of vector usually given in
linear algebra courses by the vector
space aims and we would be done but
before we do that let's try to give some
motivation behind the definition let's
start by analyzing ethologically
speaking what Vector means Vector comes
from the Latin verb V which can be
translated as to carry to move or
alternatively to
advance the word Vector was used for the
first time by William Rowan Hamilton in
the context of cians but vectors in a
particular case of Aros of two and three
dimensional cusion space have been known
under different names since the times of
the Kart Hamilton conceived a vector as
a difference of its two extreme points
for Hamilton a vector was strictly a
three-dimensional entity having three
coordinates relative to Any Given
coordinate systems and possibly both
polar or rectangular or other coordinate
systems he therefore refer to vectors as
triplets in amon's texts furthermore the
first occurrences of the word scaler can
also be found this one deres from the
Latin word Scala which means scale in a
modern sense when we talk about scaling
a vector we mean scaling the length of a
vector and possibly reflecting it if
we're multiplying it by a negative
scaler we have just two operations that
we can perform in vectors which can sum
them and we can scale them by real value
and notice that scaling a vector by real
value is nothing but a continuous
extrapolation of the discrete operation
of summing a vector and Times by itself
now we have our three-dimensional
vectors which are threedimensional
arrows in our space now we can sum them
geometrically by taking the first Vector
of the sum and position the start of the
latter with the tip of the other vector
and then drawing the Vector that goes
from the start of the second Vector to
the tip of the vector we moved and we
can scale them as we said by basically
we can think of stretching or
compressing our vector by a certain
scale or and by multiplying it by a
negative value we would result in a in a
reflected Vector so now in order to
prove some facts about these operations
we will treat them analytically and we
see that if we write our
three-dimensional arrows as strip where
the three values represent the distance
on the X Y and Z AIS with respect to the
origin then we can express analytically
the operations that we introduced
geometrically before in the following
way so now let's prove that the sum is
commutative associative and also has an
inverse and an identity so by the way we
defin it it is pretty easy to prove it
in this particular case but in the more
General case when whenever we Define an
operation like that so we take a
structure which has already well behaved
enough operation and we Define CET or
possibly NES of from of values from that
structure and then we Define a
componentwise operation between these
two PS then we will get another well-
behaved structure so in more formal
terms in this case we're talking about
groups and well since R so the real
numbers with respect to addition is a
group this this means that real addition
is commutative associative it has an
identity which is zero and every element
as an inverse and well what we the
construction we did earlier for our
triplets was nothing but basically
taking Triplets of real values and then
defining some of these triplets by
component wise sum and this is this an
algebra is called an external direct
product of groups and it is a pretty
well-known fact that it turns out that
external Direct direct product of groups
are still groups now we're going to
prove that the scalar multiplication
that we defined distributes with respect
to the addition of real
numbers now on the other hand we're
going to prove that scalar
multiplication distributes with respect
to vector
[Music]
addition furthermore if we multiply a
vector by the scalar one which is the ID
multiplicative identity of our real
numbers with respect to real number
multiplication we get back the original
Vector furthermore we want to show that
scalar multiplication that is the
multiplication of a real value by a
vector is compatible with multiplication
of real numbers now these properties are
precisely the properties that we
postulate a vector space to obey when we
study them in linear algebra the reason
why we picked these as a representative
over others might not be entirely clear
and it's fine since there is quite a lot
of eristics involved in the process this
formulation of abstract Vector spaces
which is the way in which there
presented in linear algebra textbooks
was first introduced by the Italian
mathematician jph Pano in his paper Cal
geometric theh Grassman as the title
suggests this paper was in turn deeply
influenced by grassman's work in
particular his
paper mathematic where Grassman
introduced Notions such as that of
linear Independence and dimension the
power for formalism of vector spaces and
linear algebra was thus formulated as a
consequence of the observation that
there are far more General and Abstract
objects that behave like arrows in space
this formalism was thus intuitively
induced and tically motivated by
geometry and so we can see this as a
one-way Bridge from geometrical
intuition to abstract mathematical
structures however it turns out that we
can construct Bridge the other way
around too abct Vector spaces are are an
incredibly powerful tool that allows us
to solve many geometric problems we can
define a notion of inner product that
allows us to measure lengths and angles
in our space we can export this ability
on a local level by doing it on the
tangent space of a manifold this leads
to the concept of aanan manifold we can
use quadratic forms and their Associated
matrices to study and classify conic
sections quadric surfaces and quadric
hypersurfaces and really much more can
be done but to not make this video too
long we'll stop here it is thus evident
how Vector space has form the algebraic
setting for equan geometry we now wonder
if there's an analogous construction for
hyperbolic geometry also known as laachi
geometry the answer to this question is
affirmative and the structure in
question is is known in literature with
the name of J Vector space gero Vector
spaces not only form the hyperbolic
equivalent of vector spaces for the
study of hyperbolic geometry but also
have important connections with many
areas of physics most importantly thas
procession and the velocity addition law
in special fear relativity before we
dive into the details of G Vector spaces
let's first talk about hyperbolic
geometry what is hyperbolic geometry
ukan geometry which is the usual
geometry we're all accustomed to has
some axioms in order to be treated as an
abser model these axioms are contained
in nucle element even though in most
modern books Hilbert's axium are
reported one axum in particular
tormented mathematicians for centuries
and that is the ukes fifth postulate
also known as the parallel postulate
that states that in the plane given a
line and a point not in it at most one
line parallel to the given line can be
drawn through the point and note that
this is not precisely ID's original
postulate but rather an equivalent one
commonly known as players Axiom ever
since the time of ukl mathematicians
felt like it was an a natural postate
and wanted to show that they could be
proven starting from the others however
all these attempts were not successful
this fifth axum tormented one
mathematician more than the others
yanosh b b's obsession with the latter
is shown by the following letter that
his father wrote to him you must not
attempt this approach to parallels I
know this way to the very end I have
traversed this bottom last night which
extinguished all light and joy in my
life I entreat you leave the signs of
parallels alone po however did not
surrender and in 1823 he came to the
conclusion that the fifth postulate is
independent of the others his Works were
published in 1832 independent from B the
Russian mathematician Nikolai Ivanovich
laachi came to similar conclusions they
independently Pioneer the development of
noncon geometry and this is the reason
why we often call hyperbolic geometry
laachi Bal geometry hyperbolic geometry
is not the only kind of nonukan geometry
but for this video we will focus on this
one as we said earlier hyperbolic
geometry does not respect players axium
this leads to many counterintuitive
results the first one of them is that in
hyperbolic geometry the sum of the
angles in a triangle may be less than
180° and furthermore given a line there
exists an infinite number of lines
parallel to it through a point not on
the line mathematically speaking
hyperbolic space is a reman manifold of
constant sectional curvature equal to
minus one a remanion manifold can be
intuitively understood as a space which
locally resembles RN and where we can
locally measure lengths and angles there
are various models of hyperbolic
geometry all equivalent to each other
the hyperboloid model the Beltrami
Kleine model the point ball model and
the point half space model for the
purposes of this video we will focus on
Po's model and the belic line model the
PO disc model is a model of
two-dimensional hyperbolic geometry in
which all points are inside the unit dis
in geodesics that is curves that
minimize distance are either circular
arcs contained within the disc that are
fonal to the unit circle or diameters of
the unit circle the way in which we
measure distances inside of our disc is
also different the hyperbolic distance
between two points on the disc is given
by the following formula if this formula
seems un sensical it's fine a derivation
of it starts from metric with which our
po dis model being a maranan manifold is
endowed with in the future I'll make a
video solely devoted to the PO dis in
which I will include also this
derivation we are now ready to dive into
the hortic of Juro Vector spaces Juro
Vector spaces were introduced in 1988 by
Abraham Albert uner as an attempt to
create a notion analogous to that of a
Ulan Vector space in the case of
hyperbolic geometry in particular we
will be interested in two kinds of J
Vector spaces that is mobus J Vector
spaces which will serve as an algebraic
setting for the poink AL dis model and
Einstein Jor spaces which serve on the
other hand as a setting for the Bic line
model so back to jov Vector spaces let's
see how the idea developed genetically
discovery of gur groups in juror spaces
was was gradual they emerged from the
non-commutative non-associative
algebraic structure of threedimensional
relativistically admissible velocities
with operation given by relativistic
velocity composition in his paper the
relativistic non-commutative
nonassociative group of velocities and
the toas rotation exposed how the
previously cited operation satisfies a
weak commutative and weak associative
law and he expressed them in terms of
theas rotation but what is Thomas
rotation both of these concepts are
named after the physicist liin Thomas
who noticed that in general two
successive Lawrence acceleration
Transformations do not form a Lawrence
acceleration transformation but
a len acceleration transformation
preceded or followed by space rotation
the latter is now commonly known with
the name of Fus rotation this fact was
also noticed independently by the Polish
physicist Ludwick silberstein in his
1914 textbook the F of Relativity the
following set represents in the special
Fe relativity the set of admissible
velocities relativistic velocity
composition law is given by the
following equation which as you can see
is radically different from the velocity
composition in In classical physics
which is a just a vector addition the
necessity for such an expression is due
to the fact that that we require that by
combining relativistically acceptable
velocities we need to get back another
relativistically possible velocity in
fact if we took the set of
relativistically admissible velocities
and sum them with respect to the
classical addition of vectors the
algebraic structure that we will get
will not be closed because trivially we
will be able to get velocities faster
than light by sing velocities that are
not here gamma is the Loren factor which
is pretty frequent in special
relativity by the definition of velocity
composition and admissible velocities we
gave we see that when we composed a
velocity U with a velocity V and a
velocity V with u we see that composing
U with v is different than composing V
with u so the composition that we Define
is generally
non-commutative however we see that the
two have the same magnitude from from
for this reason we can then construct a
unique rotation operator that will will
denote as Tom of u and v that is able to
quote unquote fix the non-commutativity
of our algebraic structure with the
following relation geometrically this
weak commutativity can be seen as an
action of our rotation operator Tom by
aligning these two different elements
with the same magnitude one over another
via a rotation this is the expression
for this operator and as we can see it's
quite convoluted now the following
properties
hold now at this point you might say
okay so we found an operator which quote
unquote fixes noncommutativity by
defining it at do so what well it turns
out that Thomas rotation not only fixes
the non-commutativity of our Loop of of
possible velocities but it also fixes
its non-associativity in his 1988 paper
unar realized that the thas rotation
which is usually as an isolated concept
with respect to the velocity composition
is actually an integral part of the
algebraic structure of RC this operator
which quote unquote fixed
non-commutativity and non-associativity
in our Loop of relativistically possible
velocities can then be generalized
giving rise to the notion of geration
and the abstract notion of jog group
thus just like we did with ukan Vector
spaces we took our intuition from
physics in geometry and then we
abstracted our initial structure in the
first case arrows in the second case
relativistically admissible velocities
into a more abstract model this is the
way in which Juro groups were discovered
however let's now provide another and
probably better way to way to introduce
them let's consider the complex unit dis
D we know that the complex unit dis with
the point metric is a model of
hyperbolic geometry known as the point
the most General mobile self
transformation of the unit dis assumes
the following form now if you're not
comfortable with the notion of mobious
transformation I I will leave some
resources in the description so that you
can read more about it and but for now
keep in mind that mobio Transformations
are conformal maps and furthermore M's
Transformations play a fundamental role
in complex
geometry this expression gives rise to
what we could call mobious addition
which can be geometrically interpreted
as an hyperbolic translation this
addition is neither commutative nor
associative and now we will actually
check
[Music]
it
[Music]
however inspired in the way in which we
introduced the Thomas rotation operator
in the case of Einstein velocity
addition we will introduce an operator
which we will call gation and Define it
in the following way our gation like
before repairs associativity and
commutativity laws alike this gives to
the definition of a jog
group we can also Define Joc commutative
jog groups now notice that the set of
axum that we provided is in a certain
sense quite minimalistic as you can see
we did not suppose that our identity was
unique and we only supposed that we had
a left identity in the same way we only
supposed that we had a left inverse and
we did not suppose that this left
inverse was unique we did this because
we can actually derive from these axioms
that indeed the left inverse Al must
also be the right inverse and
furthermore it must be unique and in the
same way the identity must be unique and
it must be both the left and right
identity now just like a bilon groups
with extra structure lead to the
definition of a vector space we can add
extra structure in a jur commutative
Juro group to create the notion of a
gurov vector space for hyperbolic space
now the only missing piece that we need
in order to make the unit dis in complex
plane a guro vector space as well as the
set of relativistically admissible
velocities is the concept of Jos scalar
multiplication that we also defined in
our axum for jov vactive space
but the way in which we do it is
precisely in the same way in which we
did it for geometry in particular we
take the discrete operation of summing
of actually in this case Jos summing a
element of for example the a
relativistically admissible velocity in
one case for the Einstein J Vector space
or a point and unit disc for the case of
the mobus J Vector of what will become
the mobus Jor space then by summing
itself and times you get a formula and
notice that this is unambiguous of
course the operation of addition uh of
for Loop of an element with itself is
trivially both associative and
commutative now we will get an
expression in terms of a discrete number
so in particular of a positive integer n
now if we formally substitute this
positive integer n with a real value
then we get formula for SC for Jos
scalar multiplication in our G Vector
spaces at this point we're done with the
introduction of these Concepts but
notice that you're only scraping the
surface in fact there is much more to
say about Juro Vector spaces and Juro
groups in particular about their
applications to gerot trigonometry and
for this I will leave some resources in
the description and possibly in the
future I might also make a continuation
of this video in in which we will
explore these applications with this
said I thank you for your attention and
hope that this video was
helpful
vectors if you clicked on this video you
probably already know what a vector is
however regardless on your current level
of knowledge about them we were still
going to quickly reintroduce them in a
more genetic way so what is a vector
technically we could just give the
definition of vector usually given in
linear algebra courses by the vector
space aims and we would be done but
before we do that let's try to give some
motivation behind the definition let's
start by analyzing ethologically
speaking what Vector means Vector comes
from the Latin verb V which can be
translated as to carry to move or
alternatively to
advance the word Vector was used for the
first time by William Rowan Hamilton in
the context of cians but vectors in a
particular case of Aros of two and three
dimensional cusion space have been known
under different names since the times of
the Kart Hamilton conceived a vector as
a difference of its two extreme points
for Hamilton a vector was strictly a
three-dimensional entity having three
coordinates relative to Any Given
coordinate systems and possibly both
polar or rectangular or other coordinate
systems he therefore refer to vectors as
triplets in amon's texts furthermore the
first occurrences of the word scaler can
also be found this one deres from the
Latin word Scala which means scale in a
modern sense when we talk about scaling
a vector we mean scaling the length of a
vector and possibly reflecting it if
we're multiplying it by a negative
scaler we have just two operations that
we can perform in vectors which can sum
them and we can scale them by real value
and notice that scaling a vector by real
value is nothing but a continuous
extrapolation of the discrete operation
of summing a vector and Times by itself
now we have our three-dimensional
vectors which are threedimensional
arrows in our space now we can sum them
geometrically by taking the first Vector
of the sum and position the start of the
latter with the tip of the other vector
and then drawing the Vector that goes
from the start of the second Vector to
the tip of the vector we moved and we
can scale them as we said by basically
we can think of stretching or
compressing our vector by a certain
scale or and by multiplying it by a
negative value we would result in a in a
reflected Vector so now in order to
prove some facts about these operations
we will treat them analytically and we
see that if we write our
three-dimensional arrows as strip where
the three values represent the distance
on the X Y and Z AIS with respect to the
origin then we can express analytically
the operations that we introduced
geometrically before in the following
way so now let's prove that the sum is
commutative associative and also has an
inverse and an identity so by the way we
defin it it is pretty easy to prove it
in this particular case but in the more
General case when whenever we Define an
operation like that so we take a
structure which has already well behaved
enough operation and we Define CET or
possibly NES of from of values from that
structure and then we Define a
componentwise operation between these
two PS then we will get another well-
behaved structure so in more formal
terms in this case we're talking about
groups and well since R so the real
numbers with respect to addition is a
group this this means that real addition
is commutative associative it has an
identity which is zero and every element
as an inverse and well what we the
construction we did earlier for our
triplets was nothing but basically
taking Triplets of real values and then
defining some of these triplets by
component wise sum and this is this an
algebra is called an external direct
product of groups and it is a pretty
well-known fact that it turns out that
external Direct direct product of groups
are still groups now we're going to
prove that the scalar multiplication
that we defined distributes with respect
to the addition of real
numbers now on the other hand we're
going to prove that scalar
multiplication distributes with respect
to vector
[Music]
addition furthermore if we multiply a
vector by the scalar one which is the ID
multiplicative identity of our real
numbers with respect to real number
multiplication we get back the original
Vector furthermore we want to show that
scalar multiplication that is the
multiplication of a real value by a
vector is compatible with multiplication
of real numbers now these properties are
precisely the properties that we
postulate a vector space to obey when we
study them in linear algebra the reason
why we picked these as a representative
over others might not be entirely clear
and it's fine since there is quite a lot
of eristics involved in the process this
formulation of abstract Vector spaces
which is the way in which there
presented in linear algebra textbooks
was first introduced by the Italian
mathematician jph Pano in his paper Cal
geometric theh Grassman as the title
suggests this paper was in turn deeply
influenced by grassman's work in
particular his
paper mathematic where Grassman
introduced Notions such as that of
linear Independence and dimension the
power for formalism of vector spaces and
linear algebra was thus formulated as a
consequence of the observation that
there are far more General and Abstract
objects that behave like arrows in space
this formalism was thus intuitively
induced and tically motivated by
geometry and so we can see this as a
one-way Bridge from geometrical
intuition to abstract mathematical
structures however it turns out that we
can construct Bridge the other way
around too abct Vector spaces are are an
incredibly powerful tool that allows us
to solve many geometric problems we can
define a notion of inner product that
allows us to measure lengths and angles
in our space we can export this ability
on a local level by doing it on the
tangent space of a manifold this leads
to the concept of aanan manifold we can
use quadratic forms and their Associated
matrices to study and classify conic
sections quadric surfaces and quadric
hypersurfaces and really much more can
be done but to not make this video too
long we'll stop here it is thus evident
how Vector space has form the algebraic
setting for equan geometry we now wonder
if there's an analogous construction for
hyperbolic geometry also known as laachi
geometry the answer to this question is
affirmative and the structure in
question is is known in literature with
the name of J Vector space gero Vector
spaces not only form the hyperbolic
equivalent of vector spaces for the
study of hyperbolic geometry but also
have important connections with many
areas of physics most importantly thas
procession and the velocity addition law
in special fear relativity before we
dive into the details of G Vector spaces
let's first talk about hyperbolic
geometry what is hyperbolic geometry
ukan geometry which is the usual
geometry we're all accustomed to has
some axioms in order to be treated as an
abser model these axioms are contained
in nucle element even though in most
modern books Hilbert's axium are
reported one axum in particular
tormented mathematicians for centuries
and that is the ukes fifth postulate
also known as the parallel postulate
that states that in the plane given a
line and a point not in it at most one
line parallel to the given line can be
drawn through the point and note that
this is not precisely ID's original
postulate but rather an equivalent one
commonly known as players Axiom ever
since the time of ukl mathematicians
felt like it was an a natural postate
and wanted to show that they could be
proven starting from the others however
all these attempts were not successful
this fifth axum tormented one
mathematician more than the others
yanosh b b's obsession with the latter
is shown by the following letter that
his father wrote to him you must not
attempt this approach to parallels I
know this way to the very end I have
traversed this bottom last night which
extinguished all light and joy in my
life I entreat you leave the signs of
parallels alone po however did not
surrender and in 1823 he came to the
conclusion that the fifth postulate is
independent of the others his Works were
published in 1832 independent from B the
Russian mathematician Nikolai Ivanovich
laachi came to similar conclusions they
independently Pioneer the development of
noncon geometry and this is the reason
why we often call hyperbolic geometry
laachi Bal geometry hyperbolic geometry
is not the only kind of nonukan geometry
but for this video we will focus on this
one as we said earlier hyperbolic
geometry does not respect players axium
this leads to many counterintuitive
results the first one of them is that in
hyperbolic geometry the sum of the
angles in a triangle may be less than
180° and furthermore given a line there
exists an infinite number of lines
parallel to it through a point not on
the line mathematically speaking
hyperbolic space is a reman manifold of
constant sectional curvature equal to
minus one a remanion manifold can be
intuitively understood as a space which
locally resembles RN and where we can
locally measure lengths and angles there
are various models of hyperbolic
geometry all equivalent to each other
the hyperboloid model the Beltrami
Kleine model the point ball model and
the point half space model for the
purposes of this video we will focus on
Po's model and the belic line model the
PO disc model is a model of
two-dimensional hyperbolic geometry in
which all points are inside the unit dis
in geodesics that is curves that
minimize distance are either circular
arcs contained within the disc that are
fonal to the unit circle or diameters of
the unit circle the way in which we
measure distances inside of our disc is
also different the hyperbolic distance
between two points on the disc is given
by the following formula if this formula
seems un sensical it's fine a derivation
of it starts from metric with which our
po dis model being a maranan manifold is
endowed with in the future I'll make a
video solely devoted to the PO dis in
which I will include also this
derivation we are now ready to dive into
the hortic of Juro Vector spaces Juro
Vector spaces were introduced in 1988 by
Abraham Albert uner as an attempt to
create a notion analogous to that of a
Ulan Vector space in the case of
hyperbolic geometry in particular we
will be interested in two kinds of J
Vector spaces that is mobus J Vector
spaces which will serve as an algebraic
setting for the poink AL dis model and
Einstein Jor spaces which serve on the
other hand as a setting for the Bic line
model so back to jov Vector spaces let's
see how the idea developed genetically
discovery of gur groups in juror spaces
was was gradual they emerged from the
non-commutative non-associative
algebraic structure of threedimensional
relativistically admissible velocities
with operation given by relativistic
velocity composition in his paper the
relativistic non-commutative
nonassociative group of velocities and
the toas rotation exposed how the
previously cited operation satisfies a
weak commutative and weak associative
law and he expressed them in terms of
theas rotation but what is Thomas
rotation both of these concepts are
named after the physicist liin Thomas
who noticed that in general two
successive Lawrence acceleration
Transformations do not form a Lawrence
acceleration transformation but
a len acceleration transformation
preceded or followed by space rotation
the latter is now commonly known with
the name of Fus rotation this fact was
also noticed independently by the Polish
physicist Ludwick silberstein in his
1914 textbook the F of Relativity the
following set represents in the special
Fe relativity the set of admissible
velocities relativistic velocity
composition law is given by the
following equation which as you can see
is radically different from the velocity
composition in In classical physics
which is a just a vector addition the
necessity for such an expression is due
to the fact that that we require that by
combining relativistically acceptable
velocities we need to get back another
relativistically possible velocity in
fact if we took the set of
relativistically admissible velocities
and sum them with respect to the
classical addition of vectors the
algebraic structure that we will get
will not be closed because trivially we
will be able to get velocities faster
than light by sing velocities that are
not here gamma is the Loren factor which
is pretty frequent in special
relativity by the definition of velocity
composition and admissible velocities we
gave we see that when we composed a
velocity U with a velocity V and a
velocity V with u we see that composing
U with v is different than composing V
with u so the composition that we Define
is generally
non-commutative however we see that the
two have the same magnitude from from
for this reason we can then construct a
unique rotation operator that will will
denote as Tom of u and v that is able to
quote unquote fix the non-commutativity
of our algebraic structure with the
following relation geometrically this
weak commutativity can be seen as an
action of our rotation operator Tom by
aligning these two different elements
with the same magnitude one over another
via a rotation this is the expression
for this operator and as we can see it's
quite convoluted now the following
properties
hold now at this point you might say
okay so we found an operator which quote
unquote fixes noncommutativity by
defining it at do so what well it turns
out that Thomas rotation not only fixes
the non-commutativity of our Loop of of
possible velocities but it also fixes
its non-associativity in his 1988 paper
unar realized that the thas rotation
which is usually as an isolated concept
with respect to the velocity composition
is actually an integral part of the
algebraic structure of RC this operator
which quote unquote fixed
non-commutativity and non-associativity
in our Loop of relativistically possible
velocities can then be generalized
giving rise to the notion of geration
and the abstract notion of jog group
thus just like we did with ukan Vector
spaces we took our intuition from
physics in geometry and then we
abstracted our initial structure in the
first case arrows in the second case
relativistically admissible velocities
into a more abstract model this is the
way in which Juro groups were discovered
however let's now provide another and
probably better way to way to introduce
them let's consider the complex unit dis
D we know that the complex unit dis with
the point metric is a model of
hyperbolic geometry known as the point
the most General mobile self
transformation of the unit dis assumes
the following form now if you're not
comfortable with the notion of mobious
transformation I I will leave some
resources in the description so that you
can read more about it and but for now
keep in mind that mobio Transformations
are conformal maps and furthermore M's
Transformations play a fundamental role
in complex
geometry this expression gives rise to
what we could call mobious addition
which can be geometrically interpreted
as an hyperbolic translation this
addition is neither commutative nor
associative and now we will actually
check
[Music]
it
[Music]
however inspired in the way in which we
introduced the Thomas rotation operator
in the case of Einstein velocity
addition we will introduce an operator
which we will call gation and Define it
in the following way our gation like
before repairs associativity and
commutativity laws alike this gives to
the definition of a jog
group we can also Define Joc commutative
jog groups now notice that the set of
axum that we provided is in a certain
sense quite minimalistic as you can see
we did not suppose that our identity was
unique and we only supposed that we had
a left identity in the same way we only
supposed that we had a left inverse and
we did not suppose that this left
inverse was unique we did this because
we can actually derive from these axioms
that indeed the left inverse Al must
also be the right inverse and
furthermore it must be unique and in the
same way the identity must be unique and
it must be both the left and right
identity now just like a bilon groups
with extra structure lead to the
definition of a vector space we can add
extra structure in a jur commutative
Juro group to create the notion of a
gurov vector space for hyperbolic space
now the only missing piece that we need
in order to make the unit dis in complex
plane a guro vector space as well as the
set of relativistically admissible
velocities is the concept of Jos scalar
multiplication that we also defined in
our axum for jov vactive space
but the way in which we do it is
precisely in the same way in which we
did it for geometry in particular we
take the discrete operation of summing
of actually in this case Jos summing a
element of for example the a
relativistically admissible velocity in
one case for the Einstein J Vector space
or a point and unit disc for the case of
the mobus J Vector of what will become
the mobus Jor space then by summing
itself and times you get a formula and
notice that this is unambiguous of
course the operation of addition uh of
for Loop of an element with itself is
trivially both associative and
commutative now we will get an
expression in terms of a discrete number
so in particular of a positive integer n
now if we formally substitute this
positive integer n with a real value
then we get formula for SC for Jos
scalar multiplication in our G Vector
spaces at this point we're done with the
introduction of these Concepts but
notice that you're only scraping the
surface in fact there is much more to
say about Juro Vector spaces and Juro
groups in particular about their
applications to gerot trigonometry and
for this I will leave some resources in
the description and possibly in the
future I might also make a continuation
of this video in in which we will
explore these applications with this
said I thank you for your attention and
hope that this video was
helpful
I finally got my first sponsor! Thanks Brilliant
for sponsoring the video, and head over to
brilliant.org/mathemaniac to get started for
free with their interactive lessons.
1/0 is usually undefined, but why not define
it? In some specific situations, it makes
*more* sense to define this to be infinity
than leaving it undefined. One of those situations
is Möbius maps, which are functions of the
form (az + b)/(cz + d), where a,b,c,d are
some complex constants and we are going to
impose that ad - bc is not zero. This restriction
ensures that if c is 0, d cannot possibly
be 0; otherwise ad - bc would be 0. As a result,
in this case, the denominator would never
be 0, and we have a nice linear function.
However, if c is not zero, then we have a
problem because the denominator could be 0.
This happens precisely when z = - d/c, and
in that case, we define f(-d/c) to be infinity.
However, we would also define f(infinity)
to be a/c, essentially the limit of the function
when z tends to infinity. Back to the c=0
case, we also define f(infinity), this time
the value being infinity. These different
cases can essentially be summarised as f(infinity)
is a/c, and anything non-zero divided by 0
is infinity. The reason why this condition
is required would be shown later. Now you
might be asking a question: why do we need
to torture ourselves by letting infinity into
the picture? Well, I don’t know. Mathematicians
sometimes want to torture themselves. Just
kidding, or am I? [Vsauce music]
By the end of this video, hopefully you can
see the reason why sometimes treating infinity
as a number isn’t so bad. During the course
of the video, we will first have a good intuition
of Möbius maps in 2D; then to 3D where Möbius
maps can be more naturally explained. Let’s
first focus on a special case: 1/z, or if
you prefer, a,b,c,d are 0,1,1,0 respectively.
[CHAPTER 1: The 2D perspective]
If we write z in its polar form, then its
inverse can be written down pretty easily.
Now let’s see what this really means. Negating
the argument means reflecting across the real
axis. But what about inverting the radius?
This is a classic transformation in Euclidean
geometry: inversion with respect to the unit
circle. If you watched this previous video
on Problem of Apollonius, you know some properties
of inversion, but here is a quick recap. The
official definition of inversion is that given
a point and a circle, we construct a ray from
centre to that point. Somewhere along this
line there is a point where the product of
the distances from the centre is the square
of the radius of the circle. In our case,
we are dealing with a unit circle, so the
radius squared is 1, and so the moduli really
are inverses. But this is the more general
case.
We can also invert any other object like a
circle. That means, for any point on the circle,
we can invert that point, and do the same
for all the other points. The result is still
a circle! In fact, in general, circles *always*
map to circles, if we think of straight lines
as a special kind of circle with infinite
radius. A brief explanation of this fact was
given in this video. But rather than treating
straight lines as circles with infinite radius,
think of it this way. This green line is the
image of the blue circle under inversion.
The reason why the image is not a traditional
circle is that the blue circle passes through
this red dot, the centre of this yellow circle
we are inverting with respect to.
Now what does this point invert to? By definition,
the image distance b is radius squared over
a. At the centre, the original distance a
is 0, so b would be something over 0, which
is infinity, so the image is infinitely far
away. By abuse of notation, we call that image
point infinity as well. So a circle passing
through this red centre maps to another circle
passing through the image of the red centre,
which is infinity. I know that this all sounds
ridiculous, but I promise that when we get
to 3D, everything will make a lot more sense.
Anyway, the main takeaway is that under inversion,
circles are mapped to circles, if we accept
that straight lines are just circles passing
through infinity. However, there are way simpler
transformations that have the same property.
For instance, translation obviously maps circles
to circles. Rotation also works. And reflection.
And enlargement and shrinking. So all these
transformations preserve circles, and of course
any combination of these would as well. The
combinations of all these are precisely the
Möbius maps and their conjugate, but let’s
focus on Möbius maps, i.e. why are Möbius
maps just some combinations of these? First,
we said before that the complex z^{-1} is
just a combination of reflection and geometric
inversion, so this operation does map any
circle to some other circles. Then all we
need are some algebraic manipulations. The
reason we do this is to see that Möbius maps
are just first, translation by the complex
number d/c; then by putting it in the denominator,
it is the complex inversion. Then we multiply
by some complex number, which is some stretching
or squishing and rotation. Finally, we do
another translation. All these operations
preserve circles, so Möbius maps have to
preserve circles as well.
Now since some of you want me to put in some
exercises in videos, here are something to
ponder: This decomposition only makes sense
if c is not 0, so what happens if c is 0?
In the beginning of the video, I said we need
ad - bc not equal to 0, can you see why now?
There is a final question, which is more difficult:
this shows that any Möbius map is a combination
of these five transformations , but do we
need all five? Is it possible that rotation
might just be some weird combination of translation
and inversion, or something like that? Leave
your answers in the comment below!
Anyway, when you see these transformations,
you should expect the end result to be all
circles. Otherwise, there must be something
wrong. However, the transformation looks a
bit too complicated to understand - is there
an easier way? Yes! But before we get into
that, we have to learn geometric inversion
a lot better.
[CHAPTER 2: More about inversion]
This chapter will establish several properties
of inversion, but I’m only going through
the sketch of the proofs, because ultimately
the last property is the only one that we
will need. To be honest, in general, proving
a big theorem is like going through an unfamiliar
route towards a destination. A lot of times,
proofs are either like really small step-by-step
instructions, which in some cases are really
needed, but it quickly drains your motivation;
or the proofs are way too big of a leap, like
in the memes where proofs of big theorems
are left as an exercise. The better way, I
think, is to place some reachable checkpoints
along the way, not too far or close from each
other; and more importantly, telling you ahead
of time the checkpoints you are going to get
to, and then if needed, provide the step-by-step
instruction. This will keep you motivated,
because you know what to look forward to.
Anyway, enough with the rant, this would be
the really big picture, and we will have smaller
checkpoints along the way. Don’t worry about
fine details in the first go, and feel free
to pause at any point.
First up, we have a big word here: “anticonformal”,
which means that given any two intersecting
curves, when you do the geometric inversion,
we have another pair of intersecting curves,
and importantly, the angles remain unchanged
after inversion. However, if we distinguish
the curves, like here when same-coloured curves
are images under inversion, and we assign
orientation of angles, like here measuring
from green to blue clockwise, then after inversion,
it goes from green to blue *anti*clockwise,
and so inversion reverses orientation of the
angles. Being “anticonformal” is just
a succinct way of saying both angle-preserving,
and orientation-reversing.
But why is inversion anticonformal? Ultimately,
the intersecting curves are characterised
by their tangents at intersection, so we can
just focus on this case. The good thing is
that we know what lines get mapped to, namely
circles passing through the centre. To check
that this angle is the same as the original
one, we can check that the one at the centre
is the same as the original one, but I will
leave that up to you. Next up, we have “orthogonal
circles fixed”. This essentially means that
if the blue circle intersects the yellow one
orthogonally, then inversion in the yellow
circle leaves the blue circle unchanged. The
points can map to some other points, but it
still ends up on the original circle, so the
overall figure hasn’t changed. The idea
to prove this is that the tangents to the
blue circle pass through the red centre, and
also power of a point theorem, which is, well,
a powerful theorem, but not too difficult
to prove either.
Now is the converse true as well? I.e. are
circles that are fixed orthogonal? Well, apart
from the yellow circle itself, the answer
is yes! The idea is to zoom into the intersection,
and note that the outward ray from centre
would be perpendicular here, so if the intersecting
angle is not a right angle, then points on
the left would be inverted to some point on
the left, and so the circle would not be fixed
after inversion. With all these, you can derive
that any circle passing through inverse points
has to be orthogonal to the original yellow
circle, AND that intersections of any pair
of orthogonal circles have to be inverse points.
Using this, we can even define inversion across
a line. Given a point, we can construct orthogonal
circles passing through that point, and the
other intersection should be the inverse point,
but this point is precisely the point after
reflection across the line! For the curious
among you, here is another thing to ponder:
reflection can also be seen as the limit of
inversion when the radius tends to infinity.
Can you prove this?
Anyway, that leads us to the third and final
property. As before, the yellow circle is
what we are inverting with respect to, and
the blue circle and two green points are going
to be inverted. These green points are inverse
or symmetric points with respect to the blue
circle. It turns out that after inversion,
the two points are still inverse or symmetric
to each other. This is relatively easy once
we have the previous properties. First off,
we construct circles passing through those
two points. By our previous properties, these
circles have to all be orthogonal to the blue
circle. Now invert everything with respect
to the yellow circle. Because inversion preserves
angles, these have to be right angles, and
so the intersections of these orthogonal circles
have to be inverses to each other. With this
done, we go on to 3D.
[CHAPTER 3: The 3D perspective (1/z)]
We have defined geometric inversion in 2D,
but there is nothing in 3D that stops us from
doing this. Given a sphere, points that are
inverse to each other have to lie on the same
line, and the distances also multiply to the
radius squared. Given how similar these situations
are, lots of properties that we have seen
before would generalise. In 2D, inversion
sends circles to circles, and the 3D analogue
would be sending spheres to spheres, and straight
lines would be generalised to planes, i.e.
planes are just spheres passing through infinity.
With this in mind, let’s consider this case.
This yellow sphere, as always, would be the
reference, and we want to invert this blue
sphere. Since the blue sphere passes through
the red centre, we know that it would be a
plane after inversion, but which plane? All
the points at this intersection have to be
fixed, so the plane has to pass through the
intersection, so this purple plane is the
image after inversion. But what does this
really mean? If you have a point on the sphere,
we know that the inverse has to lie on this
ray, but because we know that the image is
the plane, the intersection would be the image.
If the point is below the plane, then we again
consider the intersection of this ray, and
the plane. Even though it is phrased as spherical
inversion, this process is much more commonly
known as stereographic projection. Now, if
you haven’t watched the previous videos,
here is the concept of the Riemann sphere.
Essentially, think of the complex plane in
3D, as well as the unit sphere centred at
the origin. Then for any complex number on
the plane, we do the inverse stereographic
projection onto the sphere, so if this point
represents 1.2+1.3i, then the green point
on the sphere would correspond to the same
complex number. Under this correspondence,
the sphere would be called the Riemann sphere.
What we have said is simply that this can
be thought of as spherical inversion with
respect to this orange sphere, and so this
point should correspond to infinity, and here
comes the fun part. In the previous chapter,
we established that inversion preserves symmetric
points, so if we have two symmetric points
with respect to the yellow unit sphere, then
after inversion, these two points are still
symmetric, but across the plane, because the
plane is the image of the yellow sphere under
inversion. Let’s unpack what this means.
Being inverse points across the unit sphere
is the same as inverse points across the unit
circle on the complex plane, and doing the
inversion is the same as projecting back onto
the Riemann sphere. So on the complex plane,
inversion across the unit circle corresponds
to reflection across the equatorial plane
for the Riemann sphere! Isn’t that amazing?
Let’s put this aside for a moment. Consider
instead two symmetric points across a line.
After projecting back onto the sphere, these
two green points are symmetric to each other,
but across this vertical plane instead. So
on the complex plane, reflection across a
line corresponds to reflection across a vertical
plane for the Riemann sphere. You might have
guessed what’s coming.
Remember that 1/z is composed of reflection
across the real axis, and inversion across
the unit circle, so on the Riemann sphere,
it is a combination of reflection across the
equatorial plane, and reflection across the
vertical plane. The combined effect of these
two reflections would be rotation along the
real axis by an angle of pi. This might not
be obvious, but you would only need a bit
of thought on another vertical plane. But
the fact that 1/z corresponds to simply rotation
of the Riemann sphere should be somewhat mind-blowing!
This also means that the south pole, which
corresponds to the complex number 0, should
be rotated to the north pole, which corresponds
to infinity, so in this context it is very
natural to define 1/0 to be infinity.
[CHAPTER 4: The 3D perspective (general)]
So far, we have established that 1/z is a
rotation of the Riemann sphere. But in general,
Möbius maps also include adding and multiplying
by some constant complex numbers. Let’s
focus on translation first. This is how it
looks on the Riemann sphere, but it looks
a bit too complicated. To make it simpler,
we cheat a little bit. Instead of having a
stationary Riemann sphere, we can simply translate
the Riemann sphere itself, and stereographically
project from the north pole to get the corresponding
complex number, so that’s adding sorted.
Just like in 2D, we can translate the Riemann
sphere. What about multiplying a complex number?
For now, focus on stretching and squishing.
It turns out there is a simple cheat to represent
this: simply moving the Riemann sphere up
and down! The sphere can go up indefinitely,
but the north pole has to be higher than the
plane to do the stereographic projection.
The height of the north pole dictates the
scaling factor, the stretching or squishing.
So that’s how you can deal with multiplying
the modulus of the complex number, but what
about rotation? Turns out this is quite simple.
A rotation about the origin on the complex
plane corresponds to a rotation about the
vertical axis on the Riemann sphere. So every
simple transformation here is sorted: translations
on the complex plane are horizontal translations
of the Riemann sphere, and enlargement and
shrinking are vertical translations of the
sphere, rotation is, well, rotation on the
sphere, but about the vertical axis, and the
complex inversion is rotation about the real
axis.
Unfortunately, when you compose two functions,
the resulting transformation on the Riemann
sphere is not that straightforward. For instance,
this should be a combination of translation
and a rotation about the real axis, but it
just isn’t true. Still, it isn’t actually
awful, and in fact, in general, any Möbius
map corresponds to some rigid motion of the
Riemann sphere. So the translations and the
rotations we have identified are just simple
examples of rigid motion. And this correspondence
is 1 to 1, so any Möbius map can be *uniquely*
represented by a rigid motion. This amazing
result is established in this paper published
in 19… wait what? 2012? I’m just really
surprised that this isn’t discovered earlier,
so if you can find a paper that explicitly
demonstrates this unique correspondence earlier
than 2012, please let me know.
While Möbius maps are interesting in its
own right, they have a few important applications
or connections worth a brief mention. For
the slightly more theoretical side, the most
direct use would be in hyperbolic geometry.
Just like in Euclidean geometry where translation,
rotation and reflection and the combinations
of those are the distance-preserving transformations,
in hyperbolic geometry, the Möbius maps and
their conjugates are the distance-preserving
transformations. Möbius maps are also the
simplest examples of conformal mapping, which
are hugely important in physics, especially
fluid mechanics. But the most surprising of
all, Möbius maps are related to special relativity!
The show-off explanation is that these two
groups are isomorphic, but to understand it
more deeply, check out this book, and also
this YouTube video where the author Sir Roger
Penrose himself talked about the book.
Now you might notice that I gave a lot more
exercises in this video, because practising
is the only real way to get good with maths,
which is exactly what Brilliant is trying
to do as well. They have interactive lessons
in all the STEM subjects, with more than 60
courses to choose from. For example, if you
want to learn more about the power of a point
theorem I mentioned before, you can do so
in their Geometry II course; or if you want
to understand the relationship between special
relativity and Möbius maps, you can learn
more about special relativity in Brilliant.
This is the bit about causality, but the course
can get quite deep, all the way to 4-vectors
and 4-momentum conservation. To sign up for
Brilliant, head over to brilliant.org/mathemaniac
to get started *for free*, and the first 200
viewers will get 20% off their annual membership.
Thanks Brilliant for sponsoring this video,
and the patrons for supporting the channel.
If you enjoyed this video, don’t forget
to subscribe with notifications on, and like,
comment and share the video as well. See you
next time!
I finally got my first sponsor! Thanks Brilliant
for sponsoring the video, and head over to
brilliant.org/mathemaniac to get started for
free with their interactive lessons.
1/0 is usually undefined, but why not define
it? In some specific situations, it makes
*more* sense to define this to be infinity
than leaving it undefined. One of those situations
is Möbius maps, which are functions of the
form (az + b)/(cz + d), where a,b,c,d are
some complex constants and we are going to
impose that ad - bc is not zero. This restriction
ensures that if c is 0, d cannot possibly
be 0; otherwise ad - bc would be 0. As a result,
in this case, the denominator would never
be 0, and we have a nice linear function.
However, if c is not zero, then we have a
problem because the denominator could be 0.
This happens precisely when z = - d/c, and
in that case, we define f(-d/c) to be infinity.
However, we would also define f(infinity)
to be a/c, essentially the limit of the function
when z tends to infinity. Back to the c=0
case, we also define f(infinity), this time
the value being infinity. These different
cases can essentially be summarised as f(infinity)
is a/c, and anything non-zero divided by 0
is infinity. The reason why this condition
is required would be shown later. Now you
might be asking a question: why do we need
to torture ourselves by letting infinity into
the picture? Well, I don’t know. Mathematicians
sometimes want to torture themselves. Just
kidding, or am I? [Vsauce music]
By the end of this video, hopefully you can
see the reason why sometimes treating infinity
as a number isn’t so bad. During the course
of the video, we will first have a good intuition
of Möbius maps in 2D; then to 3D where Möbius
maps can be more naturally explained. Let’s
first focus on a special case: 1/z, or if
you prefer, a,b,c,d are 0,1,1,0 respectively.
[CHAPTER 1: The 2D perspective]
If we write z in its polar form, then its
inverse can be written down pretty easily.
Now let’s see what this really means. Negating
the argument means reflecting across the real
axis. But what about inverting the radius?
This is a classic transformation in Euclidean
geometry: inversion with respect to the unit
circle. If you watched this previous video
on Problem of Apollonius, you know some properties
of inversion, but here is a quick recap. The
official definition of inversion is that given
a point and a circle, we construct a ray from
centre to that point. Somewhere along this
line there is a point where the product of
the distances from the centre is the square
of the radius of the circle. In our case,
we are dealing with a unit circle, so the
radius squared is 1, and so the moduli really
are inverses. But this is the more general
case.
We can also invert any other object like a
circle. That means, for any point on the circle,
we can invert that point, and do the same
for all the other points. The result is still
a circle! In fact, in general, circles *always*
map to circles, if we think of straight lines
as a special kind of circle with infinite
radius. A brief explanation of this fact was
given in this video. But rather than treating
straight lines as circles with infinite radius,
think of it this way. This green line is the
image of the blue circle under inversion.
The reason why the image is not a traditional
circle is that the blue circle passes through
this red dot, the centre of this yellow circle
we are inverting with respect to.
Now what does this point invert to? By definition,
the image distance b is radius squared over
a. At the centre, the original distance a
is 0, so b would be something over 0, which
is infinity, so the image is infinitely far
away. By abuse of notation, we call that image
point infinity as well. So a circle passing
through this red centre maps to another circle
passing through the image of the red centre,
which is infinity. I know that this all sounds
ridiculous, but I promise that when we get
to 3D, everything will make a lot more sense.
Anyway, the main takeaway is that under inversion,
circles are mapped to circles, if we accept
that straight lines are just circles passing
through infinity. However, there are way simpler
transformations that have the same property.
For instance, translation obviously maps circles
to circles. Rotation also works. And reflection.
And enlargement and shrinking. So all these
transformations preserve circles, and of course
any combination of these would as well. The
combinations of all these are precisely the
Möbius maps and their conjugate, but let’s
focus on Möbius maps, i.e. why are Möbius
maps just some combinations of these? First,
we said before that the complex z^{-1} is
just a combination of reflection and geometric
inversion, so this operation does map any
circle to some other circles. Then all we
need are some algebraic manipulations. The
reason we do this is to see that Möbius maps
are just first, translation by the complex
number d/c; then by putting it in the denominator,
it is the complex inversion. Then we multiply
by some complex number, which is some stretching
or squishing and rotation. Finally, we do
another translation. All these operations
preserve circles, so Möbius maps have to
preserve circles as well.
Now since some of you want me to put in some
exercises in videos, here are something to
ponder: This decomposition only makes sense
if c is not 0, so what happens if c is 0?
In the beginning of the video, I said we need
ad - bc not equal to 0, can you see why now?
There is a final question, which is more difficult:
this shows that any Möbius map is a combination
of these five transformations , but do we
need all five? Is it possible that rotation
might just be some weird combination of translation
and inversion, or something like that? Leave
your answers in the comment below!
Anyway, when you see these transformations,
you should expect the end result to be all
circles. Otherwise, there must be something
wrong. However, the transformation looks a
bit too complicated to understand - is there
an easier way? Yes! But before we get into
that, we have to learn geometric inversion
a lot better.
[CHAPTER 2: More about inversion]
This chapter will establish several properties
of inversion, but I’m only going through
the sketch of the proofs, because ultimately
the last property is the only one that we
will need. To be honest, in general, proving
a big theorem is like going through an unfamiliar
route towards a destination. A lot of times,
proofs are either like really small step-by-step
instructions, which in some cases are really
needed, but it quickly drains your motivation;
or the proofs are way too big of a leap, like
in the memes where proofs of big theorems
are left as an exercise. The better way, I
think, is to place some reachable checkpoints
along the way, not too far or close from each
other; and more importantly, telling you ahead
of time the checkpoints you are going to get
to, and then if needed, provide the step-by-step
instruction. This will keep you motivated,
because you know what to look forward to.
Anyway, enough with the rant, this would be
the really big picture, and we will have smaller
checkpoints along the way. Don’t worry about
fine details in the first go, and feel free
to pause at any point.
First up, we have a big word here: “anticonformal”,
which means that given any two intersecting
curves, when you do the geometric inversion,
we have another pair of intersecting curves,
and importantly, the angles remain unchanged
after inversion. However, if we distinguish
the curves, like here when same-coloured curves
are images under inversion, and we assign
orientation of angles, like here measuring
from green to blue clockwise, then after inversion,
it goes from green to blue *anti*clockwise,
and so inversion reverses orientation of the
angles. Being “anticonformal” is just
a succinct way of saying both angle-preserving,
and orientation-reversing.
But why is inversion anticonformal? Ultimately,
the intersecting curves are characterised
by their tangents at intersection, so we can
just focus on this case. The good thing is
that we know what lines get mapped to, namely
circles passing through the centre. To check
that this angle is the same as the original
one, we can check that the one at the centre
is the same as the original one, but I will
leave that up to you. Next up, we have “orthogonal
circles fixed”. This essentially means that
if the blue circle intersects the yellow one
orthogonally, then inversion in the yellow
circle leaves the blue circle unchanged. The
points can map to some other points, but it
still ends up on the original circle, so the
overall figure hasn’t changed. The idea
to prove this is that the tangents to the
blue circle pass through the red centre, and
also power of a point theorem, which is, well,
a powerful theorem, but not too difficult
to prove either.
Now is the converse true as well? I.e. are
circles that are fixed orthogonal? Well, apart
from the yellow circle itself, the answer
is yes! The idea is to zoom into the intersection,
and note that the outward ray from centre
would be perpendicular here, so if the intersecting
angle is not a right angle, then points on
the left would be inverted to some point on
the left, and so the circle would not be fixed
after inversion. With all these, you can derive
that any circle passing through inverse points
has to be orthogonal to the original yellow
circle, AND that intersections of any pair
of orthogonal circles have to be inverse points.
Using this, we can even define inversion across
a line. Given a point, we can construct orthogonal
circles passing through that point, and the
other intersection should be the inverse point,
but this point is precisely the point after
reflection across the line! For the curious
among you, here is another thing to ponder:
reflection can also be seen as the limit of
inversion when the radius tends to infinity.
Can you prove this?
Anyway, that leads us to the third and final
property. As before, the yellow circle is
what we are inverting with respect to, and
the blue circle and two green points are going
to be inverted. These green points are inverse
or symmetric points with respect to the blue
circle. It turns out that after inversion,
the two points are still inverse or symmetric
to each other. This is relatively easy once
we have the previous properties. First off,
we construct circles passing through those
two points. By our previous properties, these
circles have to all be orthogonal to the blue
circle. Now invert everything with respect
to the yellow circle. Because inversion preserves
angles, these have to be right angles, and
so the intersections of these orthogonal circles
have to be inverses to each other. With this
done, we go on to 3D.
[CHAPTER 3: The 3D perspective (1/z)]
We have defined geometric inversion in 2D,
but there is nothing in 3D that stops us from
doing this. Given a sphere, points that are
inverse to each other have to lie on the same
line, and the distances also multiply to the
radius squared. Given how similar these situations
are, lots of properties that we have seen
before would generalise. In 2D, inversion
sends circles to circles, and the 3D analogue
would be sending spheres to spheres, and straight
lines would be generalised to planes, i.e.
planes are just spheres passing through infinity.
With this in mind, let’s consider this case.
This yellow sphere, as always, would be the
reference, and we want to invert this blue
sphere. Since the blue sphere passes through
the red centre, we know that it would be a
plane after inversion, but which plane? All
the points at this intersection have to be
fixed, so the plane has to pass through the
intersection, so this purple plane is the
image after inversion. But what does this
really mean? If you have a point on the sphere,
we know that the inverse has to lie on this
ray, but because we know that the image is
the plane, the intersection would be the image.
If the point is below the plane, then we again
consider the intersection of this ray, and
the plane. Even though it is phrased as spherical
inversion, this process is much more commonly
known as stereographic projection. Now, if
you haven’t watched the previous videos,
here is the concept of the Riemann sphere.
Essentially, think of the complex plane in
3D, as well as the unit sphere centred at
the origin. Then for any complex number on
the plane, we do the inverse stereographic
projection onto the sphere, so if this point
represents 1.2+1.3i, then the green point
on the sphere would correspond to the same
complex number. Under this correspondence,
the sphere would be called the Riemann sphere.
What we have said is simply that this can
be thought of as spherical inversion with
respect to this orange sphere, and so this
point should correspond to infinity, and here
comes the fun part. In the previous chapter,
we established that inversion preserves symmetric
points, so if we have two symmetric points
with respect to the yellow unit sphere, then
after inversion, these two points are still
symmetric, but across the plane, because the
plane is the image of the yellow sphere under
inversion. Let’s unpack what this means.
Being inverse points across the unit sphere
is the same as inverse points across the unit
circle on the complex plane, and doing the
inversion is the same as projecting back onto
the Riemann sphere. So on the complex plane,
inversion across the unit circle corresponds
to reflection across the equatorial plane
for the Riemann sphere! Isn’t that amazing?
Let’s put this aside for a moment. Consider
instead two symmetric points across a line.
After projecting back onto the sphere, these
two green points are symmetric to each other,
but across this vertical plane instead. So
on the complex plane, reflection across a
line corresponds to reflection across a vertical
plane for the Riemann sphere. You might have
guessed what’s coming.
Remember that 1/z is composed of reflection
across the real axis, and inversion across
the unit circle, so on the Riemann sphere,
it is a combination of reflection across the
equatorial plane, and reflection across the
vertical plane. The combined effect of these
two reflections would be rotation along the
real axis by an angle of pi. This might not
be obvious, but you would only need a bit
of thought on another vertical plane. But
the fact that 1/z corresponds to simply rotation
of the Riemann sphere should be somewhat mind-blowing!
This also means that the south pole, which
corresponds to the complex number 0, should
be rotated to the north pole, which corresponds
to infinity, so in this context it is very
natural to define 1/0 to be infinity.
[CHAPTER 4: The 3D perspective (general)]
So far, we have established that 1/z is a
rotation of the Riemann sphere. But in general,
Möbius maps also include adding and multiplying
by some constant complex numbers. Let’s
focus on translation first. This is how it
looks on the Riemann sphere, but it looks
a bit too complicated. To make it simpler,
we cheat a little bit. Instead of having a
stationary Riemann sphere, we can simply translate
the Riemann sphere itself, and stereographically
project from the north pole to get the corresponding
complex number, so that’s adding sorted.
Just like in 2D, we can translate the Riemann
sphere. What about multiplying a complex number?
For now, focus on stretching and squishing.
It turns out there is a simple cheat to represent
this: simply moving the Riemann sphere up
and down! The sphere can go up indefinitely,
but the north pole has to be higher than the
plane to do the stereographic projection.
The height of the north pole dictates the
scaling factor, the stretching or squishing.
So that’s how you can deal with multiplying
the modulus of the complex number, but what
about rotation? Turns out this is quite simple.
A rotation about the origin on the complex
plane corresponds to a rotation about the
vertical axis on the Riemann sphere. So every
simple transformation here is sorted: translations
on the complex plane are horizontal translations
of the Riemann sphere, and enlargement and
shrinking are vertical translations of the
sphere, rotation is, well, rotation on the
sphere, but about the vertical axis, and the
complex inversion is rotation about the real
axis.
Unfortunately, when you compose two functions,
the resulting transformation on the Riemann
sphere is not that straightforward. For instance,
this should be a combination of translation
and a rotation about the real axis, but it
just isn’t true. Still, it isn’t actually
awful, and in fact, in general, any Möbius
map corresponds to some rigid motion of the
Riemann sphere. So the translations and the
rotations we have identified are just simple
examples of rigid motion. And this correspondence
is 1 to 1, so any Möbius map can be *uniquely*
represented by a rigid motion. This amazing
result is established in this paper published
in 19… wait what? 2012? I’m just really
surprised that this isn’t discovered earlier,
so if you can find a paper that explicitly
demonstrates this unique correspondence earlier
than 2012, please let me know.
While Möbius maps are interesting in its
own right, they have a few important applications
or connections worth a brief mention. For
the slightly more theoretical side, the most
direct use would be in hyperbolic geometry.
Just like in Euclidean geometry where translation,
rotation and reflection and the combinations
of those are the distance-preserving transformations,
in hyperbolic geometry, the Möbius maps and
their conjugates are the distance-preserving
transformations. Möbius maps are also the
simplest examples of conformal mapping, which
are hugely important in physics, especially
fluid mechanics. But the most surprising of
all, Möbius maps are related to special relativity!
The show-off explanation is that these two
groups are isomorphic, but to understand it
more deeply, check out this book, and also
this YouTube video where the author Sir Roger
Penrose himself talked about the book.
Now you might notice that I gave a lot more
exercises in this video, because practising
is the only real way to get good with maths,
which is exactly what Brilliant is trying
to do as well. They have interactive lessons
in all the STEM subjects, with more than 60
courses to choose from. For example, if you
want to learn more about the power of a point
theorem I mentioned before, you can do so
in their Geometry II course; or if you want
to understand the relationship between special
relativity and Möbius maps, you can learn
more about special relativity in Brilliant.
This is the bit about causality, but the course
can get quite deep, all the way to 4-vectors
and 4-momentum conservation. To sign up for
Brilliant, head over to brilliant.org/mathemaniac
to get started *for free*, and the first 200
viewers will get 20% off their annual membership.
Thanks Brilliant for sponsoring this video,
and the patrons for supporting the channel.
If you enjoyed this video, don’t forget
to subscribe with notifications on, and like,
comment and share the video as well. See you
next time!

