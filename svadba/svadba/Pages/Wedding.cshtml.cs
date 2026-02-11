using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Mvc.RazorPages;

namespace svadba.Pages
{
    public class WeddingModel : PageModel
    {
        public void OnGet()
        {
        }

        public IActionResult OnPost(string name, string email, string guests, string attending, string message)
        {
            // TODO: Обработка RSVP формы
            // Можно отправить email или сохранить в БД
            return RedirectToPage("/Wedding");
        }
    }
}
