(ns fake-news-classifier.core
  (:require [reagent.core :as r]
            ["react-dom/client" :as ReactDOMClient]
            [lambdaisland.fetch :as http]
            [cljs.core :as c]))

(defonce container (.getElementById js/document "app"))
(defonce root (.createRoot ReactDOMClient container))

(def ^:private api-gateway-url "https://d09r3jvmpj.execute-api.us-west-1.amazonaws.com/prod")

(def app-state (r/atom {:headline ""
                        :prediction-result nil
                        :loading? false
                        :error-message nil}))

(defn fetch-prediction []
  (let [headline (:headline @app-state)
        url (str api-gateway-url "/predict")]
    (when (not-empty headline)
      (r/rswap! app-state assoc :loading? true :error-message nil)
      (-> (http/post url
                     {:body {:headline headline}
                      :content-type :json
                      :headers {"Content-Type" "application/json"}})
          (.then (fn [response]
                   (let [body (-> response :body (js->clj :keywordize-keys true))
                         status (:status response)]
                     (if (<= 200 status 299)
                       (r/rswap! app-state assoc
                                 :prediction-result body
                                 :loading? false)
                       (throw (js/Error. (str "HTTP Error " status " - " (pr-str body))))))))
          (.catch (fn [error]
                    (js/console.error "Fetch error:" error)
                    (r/rswap! app-state assoc
                              :error-message (str "Fetch Error: " (.-message error))
                              :loading? false)))))))

(defn headline-input []
  [:div.mb-4
   [:label.block.text-gray-700.text-sm.font-bold.mb-2 {:for "headline"} "News Headline:"]
   [:input#headline.shadow.appearance-none.border.rounded.w-full.py-2.px-3.text-gray-700.leading-tight.focus:outline-none.focus:shadow-outline
    {:type "text"
     :placeholder "Enter a news headline..."
     :value (:headline @app-state)
     :on-change #(r/rswap! app-state assoc :headline (-> % .-target .-value))}]])

(defn predict-button []
  [:button.bg-blue-500.hover:bg-blue-700.text-white.font-bold.py-2.px-4.rounded.focus:outline-none.focus:shadow-outline
   {:on-click fetch-prediction
    :disabled (:loading? @app-state)}
   (if (:loading? @app-state) "Predicting..." "Predict Fake News")])

(defn prediction-display []
  (let [{:keys [prediction-result error-message]} @app-state]
    [:div.mt-6
     (when error-message
       [:div.bg-red-100.border.border-red-400.text-red-700.px-4.py-3.rounded.relative {:role "alert"}
        [:strong.font-bold "Error: "]
        [:span.block.sm:inline error-message]])

     (when prediction-result
       [:div.bg-gray-100.p-4.rounded-lg.shadow-md
        [:h3.text-lg.font-semibold.mb-2 "Prediction Result:"]
        [:p.mb-1 [:strong "Headline: "] (:headline prediction-result)]
        [:p.mb-1 [:strong "Message: "] (:message prediction-result)]])]))

(defn app-root []
  [:div.container.mx-auto.p-4.max-w-xl.bg-white.rounded-lg.shadow-xl.mt-10
   [:h1.text-3xl.font-bold.mb-6.text-center.text-gray-800 "Fake News Classifier"]
   [headline-input]
   [predict-button]
   [prediction-display]])

(defn init []
  (.render root (r/as-element [app-root])))

(defn ^:after-load on-reload []
  (init))
